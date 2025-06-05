import sys
import os
import threading
import tempfile
import sounddevice as sd
import numpy as np
import wavio
import whisper
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from deep_translator import GoogleTranslator
from PyQt6 import QtCore, QtGui, QtWidgets
from difflib import SequenceMatcher

RECIPES = [
    {"name": "Sałatka grecka", "ingredients": ["pomidor", "ogórek", "cebula", "ser", "oliwki"]},
    {"name": "Jajecznica", "ingredients": ["jajko", "masło", "cebula"]},
    {"name": "Spaghetti Bolognese", "ingredients": ["makaron", "mięso mielone", "pomidor", "cebula", "czosnek"]},
    {"name": "Kanapka z serem", "ingredients": ["chleb", "ser", "masło"]},
    {"name": "Zupa pomidorowa", "ingredients": ["pomidor", "cebula", "czosnek", "marchew"]},
    {"name": "Kurczak pieczony", "ingredients": ["kurczak", "czosnek", "papryka"]},
    {"name": "Placki ziemniaczane", "ingredients": ["ziemniak", "cebula", "jajko", "mąka"]},
    {"name": "Omlet warzywny", "ingredients": ["jajko", "papryka", "cebula", "pomidor"]},
    {"name": "Pierogi ruskie", "ingredients": ["ziemniak", "ser biały", "cebula", "mąka"]},
    {"name": "Sałatka jarzynowa", "ingredients": ["marchew", "groszek", "ziemniak", "jajko", "majonez"]},
    {"name": "Ryba smażona", "ingredients": ["ryba", "mąka", "olej"]},
    {"name": "Gulasz wołowy", "ingredients": ["wołowina", "cebula", "papryka", "czosnek"]},
    {"name": "Risotto", "ingredients": ["ryż", "cebula", "parmezan", "masło"]},
    {"name": "Makaron z pesto", "ingredients": ["makaron", "bazylia", "czosnek", "ser"]},
    {"name": "Sałatka owocowa", "ingredients": ["jabłko", "banan", "pomarańcza", "kiwi"]},
    {"name": "Zupa ogórkowa", "ingredients": ["ogórek kiszony", "ziemniak", "marchew", "śmietana"]},
    {"name": "Pizza Margherita", "ingredients": ["mąka", "pomidor", "ser", "bazylia"]},
    {"name": "Chili con carne", "ingredients": ["fasola", "mięso mielone", "papryka", "cebula", "czosnek"]},
    {"name": "Kotlet schabowy", "ingredients": ["schab", "jajko", "bułka tarta", "olej"]},
    {"name": "Sałatka z tuńczykiem", "ingredients": ["tuńczyk", "sałata", "pomidor", "cebula"]},
    {"name": "Zupa krem z dyni", "ingredients": ["dynia", "cebula", "śmietana"]},
    {"name": "Leczo", "ingredients": ["papryka", "cukinia", "kiełbasa", "cebula", "pomidor"]},
    {"name": "Naleśniki z serem", "ingredients": ["mąka", "jajko", "ser", "mleko"]},
    {"name": "Sałatka Cezar", "ingredients": ["sałata", "kurczak", "ser", "grzanki"]},
    {"name": "Zupa grochowa", "ingredients": ["groch", "cebula", "marchew", "czosnek"]},
]

def filter_recipes(ingredients):
    if not ingredients:
        return []
    filtered = []
    for recipe in RECIPES:
        if all(any(ing.lower() in ri.lower() or ri.lower() in ing.lower() for ri in recipe['ingredients']) for ing in ingredients):
            filtered.append(recipe)
    return filtered

def is_similar(a, b, threshold=0.8):
        return SequenceMatcher(None, a, b).ratio() >= threshold

def translate_words_separately(text, source_lang):
    words = text.strip().split()
    translated_words = []
    try:
        translator = GoogleTranslator(source=source_lang, target='pl')
    except Exception:
        return text  # fallback bez tłumaczenia
    for w in words:
        try:
            tw = translator.translate(w)
            translated_words.append(tw)
        except Exception:
            translated_words.append(w)
    return " ".join(translated_words)

class RecorderThread(QtCore.QThread):
    recording_finished = QtCore.pyqtSignal(str)

    def run(self):
        duration = 5
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        filename = os.path.join(tempfile.gettempdir(), "audio.wav")
        wavio.write(filename, recording, fs, sampwidth=2)
        self.recording_finished.emit(filename)

class TranscribeThread(QtCore.QThread):
    transcription_done = QtCore.pyqtSignal(str, str, list)

    def __init__(self, audio_path, model_name, forced_lang):
        super().__init__()
        self.audio_path = audio_path
        self.model_name = model_name
        self.forced_lang = forced_lang  # może być "auto" lub np. "pl"

    def run(self):
        model = whisper.load_model(self.model_name)
        options = {}
        if self.forced_lang != "auto":
            # wymuszamy język przy transkrypcji (jeśli Whisper obsługuje)
            options["language"] = self.forced_lang
            options["task"] = "transcribe"

        result = model.transcribe(self.audio_path, **options, fp16=False)
        text = result["text"].strip()
        lang = result.get("language", "unknown")

        if self.forced_lang != "pl":
            try:
                translated = translate_words_separately(text, lang if self.forced_lang == "auto" else self.forced_lang)
            except Exception:
                translated = text
        else:
            translated = text

        ingredients_list = [ing for rec in RECIPES for ing in rec['ingredients']]
        ingredients_list = list(set(ingredients_list))
        words = translated.lower().split()
        detected_ingredients = []
        
        for w in words:
            for ing in ingredients_list:
                if is_similar(w, ing):
                    if ing not in detected_ingredients:
                        detected_ingredients.append(ing)


            self.transcription_done.emit(text, translated, detected_ingredients)

class StartWindow(QtWidgets.QWidget):
    switch_to_result = QtCore.pyqtSignal(str, str, list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Filtrowanie przepisów na podstawie mowy")
        self.layout = QtWidgets.QVBoxLayout(self)

        self.label_info = QtWidgets.QLabel("🎙️ Powiedz składniki potrawy lub wczytaj plik audio.")
        self.layout.addWidget(self.label_info)

        self.btn_load = QtWidgets.QPushButton("Wczytaj plik audio")
        self.btn_load.clicked.connect(self.load_audio)
        self.layout.addWidget(self.btn_load)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("large")
        self.layout.addWidget(QtWidgets.QLabel("Wybierz model Whisper:"))
        self.layout.addWidget(self.model_combo)

        # NOWOŚĆ: wybór języka źródłowego
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItem("Wykryj język", userData="auto")
        self.lang_combo.addItem("Polski", userData="pl")
        self.lang_combo.addItem("Angielski", userData="en")
        self.lang_combo.addItem("Niemiecki", userData="de")
        self.lang_combo.addItem("Francuski", userData="fr")
        self.lang_combo.addItem("Hiszpański", userData="es")
        self.lang_combo.setCurrentIndex(0)
        self.layout.addWidget(QtWidgets.QLabel("Wybierz język nagrania:"))
        self.layout.addWidget(self.lang_combo)

        self.btn_record = QtWidgets.QPushButton()
        self.btn_record.setFixedSize(100, 100)
        self.btn_record.setIcon(QtGui.QIcon.fromTheme("microphone-sensitivity-high"))
        self.btn_record.setIconSize(QtCore.QSize(64, 64))
        self.btn_record.setStyleSheet("""
            QPushButton {
                border-radius: 50px;
                background-color: #4285F4;
                color: white;
                font-size: 24px;
            }
            QPushButton:pressed {
                background-color: #3367D6;
            }
        """)
        self.btn_record.clicked.connect(self.record_audio)
        self.layout.addWidget(self.btn_record, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximum(5000)
        self.progress.setMinimum(0)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.layout.addWidget(self.progress)

        self.record_start_time = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_progress)

        self.record_thread = None
        self.transcribe_thread = None

    def load_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Wczytaj plik audio", "", "Audio Files (*.wav *.mp3 *.m4a)")
        if path:
            self.transcribe(path)

    def record_audio(self):
        self.btn_record.setEnabled(False)
        self.progress.setValue(5000)
        self.progress.show()
        self.record_start_time = QtCore.QTime.currentTime()
        self.timer.start(50)

        self.record_thread = RecorderThread()
        self.record_thread.recording_finished.connect(self.on_recording_finished)
        self.record_thread.start()

    def update_progress(self):
        elapsed = self.record_start_time.msecsTo(QtCore.QTime.currentTime())
        remaining = max(0, 5000 - elapsed)
        self.progress.setValue(remaining)
        if remaining == 0:
            self.timer.stop()
            self.progress.hide()

    def on_recording_finished(self, filename):
        self.btn_record.setEnabled(True)
        self.progress.hide()
        self.transcribe(filename)

    def transcribe(self, filepath):
        model_name = self.model_combo.currentText()
        forced_lang = self.lang_combo.currentData()
        self.label_info.setText(f"🧠 Ładowanie modelu Whisper '{model_name}', transkrypcja...")
        self.transcribe_thread = TranscribeThread(filepath, model_name, forced_lang)
        self.transcribe_thread.transcription_done.connect(self.on_transcription_done)
        self.transcribe_thread.start()

    def on_transcription_done(self, orig_text, translated_text, ingredients):
        self.switch_to_result.emit(orig_text, translated_text, ingredients)

class ResultWindow(QtWidgets.QWidget):
    switch_to_start = QtCore.pyqtSignal()
    switch_to_recipes = QtCore.pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wynik rozpoznawania")
        self.layout = QtWidgets.QVBoxLayout(self)

        self.orig_label = QtWidgets.QLabel("Tekst oryginalny:")
        self.layout.addWidget(self.orig_label)
        self.orig_text = QtWidgets.QLabel("")
        self.layout.addWidget(self.orig_text)

        self.translated_label = QtWidgets.QLabel("Przetłumaczony tekst:")
        self.layout.addWidget(self.translated_label)
        self.translated_text = QtWidgets.QLabel("")
        self.layout.addWidget(self.translated_text)

        self.ingredients_label = QtWidgets.QLabel("Wykryte składniki:")
        self.layout.addWidget(self.ingredients_label)
        self.ingredients_text = QtWidgets.QLabel("")
        self.layout.addWidget(self.ingredients_text)

        hbox = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton()
        self.btn_back.setIcon(QtGui.QIcon.fromTheme("go-previous"))
        self.btn_back.setToolTip("Wróć do nagrywania")
        self.btn_back.clicked.connect(lambda: self.switch_to_start.emit())
        hbox.addWidget(self.btn_back)

        self.btn_show_recipes = QtWidgets.QPushButton()
        self.btn_show_recipes.setIcon(QtGui.QIcon.fromTheme("view-list"))
        self.btn_show_recipes.setToolTip("Pokaż przepisy")
        self.btn_show_recipes.clicked.connect(self.show_recipes)
        hbox.addWidget(self.btn_show_recipes)

        self.layout.addLayout(hbox)

        self.detected_ingredients = []

    def show_result(self, orig_text, translated_text, ingredients):
        self.orig_text.setText(orig_text)
        self.translated_text.setText(translated_text)
        self.ingredients_text.setText(", ".join(ingredients) if ingredients else "Brak wykrytych składników")
        self.detected_ingredients = ingredients

    def show_recipes(self):
        self.switch_to_recipes.emit(self.detected_ingredients)

class RecipesWindow(QtWidgets.QWidget):
    switch_to_start = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Przepisy")
        self.layout = QtWidgets.QVBoxLayout(self)

        self.list_widget = QtWidgets.QListWidget()
        self.layout.addWidget(self.list_widget)

        self.btn_back = QtWidgets.QPushButton("Wróć")
        self.btn_back.clicked.connect(lambda: self.switch_to_start.emit())
        self.layout.addWidget(self.btn_back)

    def show_recipes(self, ingredients):
        self.list_widget.clear()
        filtered = filter_recipes(ingredients)
        if filtered:
            for r in filtered:
                self.list_widget.addItem(f"{r['name']} (składniki: {', '.join(r['ingredients'])})")
        else:
            self.list_widget.addItem("⚠️ Brak przepisów spełniających wymagania.")

class MainWindow(QtWidgets.QStackedWidget):
    def __init__(self):
        super().__init__()
        self.start_win = StartWindow()
        self.result_win = ResultWindow()
        self.recipes_win = RecipesWindow()

        self.addWidget(self.start_win)
        self.addWidget(self.result_win)
        self.addWidget(self.recipes_win)

        self.start_win.switch_to_result.connect(self.show_result)
        self.result_win.switch_to_start.connect(self.show_start)
        self.result_win.switch_to_recipes.connect(self.show_recipes)
        self.recipes_win.switch_to_start.connect(self.show_start)

        self.setWindowTitle("Filtrowanie przepisów - Mowa")
        self.setFixedSize(500, 400)
        self.show_start()

    def show_start(self):
        self.setCurrentWidget(self.start_win)

    def show_result(self, orig_text, translated_text, ingredients):
        self.result_win.show_result(orig_text, translated_text, ingredients)
        self.setCurrentWidget(self.result_win)

    def show_recipes(self, ingredients):
        self.recipes_win.show_recipes(ingredients)
        self.setCurrentWidget(self.recipes_win)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
