"""Text Autocomplete System: PyQt6 front end.

Typing is debounced: suggestions are computed once the user pauses for
DEBOUNCE_MS, so a burst of keystrokes costs one lookup. The list shows the
ten most frequent completions of the word being typed; clicking one
completes the word and teaches the ranking (see autocomplete/learning.py).
When no word starts with what was typed, close spellings are offered.
"""
import os
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QLineEdit, QListWidget, QVBoxLayout, QWidget

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autocomplete import Autocompleter  # noqa: E402

DEBOUNCE_MS = 60


def last_word(text):
    """The word under construction, or '' when the text ends with a space."""
    if not text or text[-1].isspace():
        return ""
    return text.split()[-1]


class AutocompleteApp(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self.refresh)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Autocomplete System")
        self.setGeometry(100, 100, 400, 300)
        layout = QVBoxLayout()
        self.label = QLabel("Start typing to get suggestions:")
        layout.addWidget(self.label)
        self.input = QLineEdit()
        self.input.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.input)
        self.suggestions = QListWidget()
        self.suggestions.itemClicked.connect(self.on_suggestion_clicked)
        layout.addWidget(self.suggestions)
        self.status = QLabel("")
        layout.addWidget(self.status)
        self.setLayout(layout)

    def on_text_changed(self):
        self.timer.start()  # restart the pause timer on every keystroke

    def refresh(self):
        self.suggestions.clear()
        prefix = last_word(self.input.text())
        words, is_fuzzy = self.engine.suggest(prefix)
        self.suggestions.addItems(words)
        self.status.setText("Did you mean:" if is_fuzzy and words else "")

    def on_suggestion_clicked(self, item):
        word = item.text()
        words = self.input.text().split()
        if words:
            words[-1] = word
        else:
            words = [word]
        self.engine.accept(word)
        self.timer.stop()
        self.input.blockSignals(True)
        self.input.setText(" ".join(words) + " ")
        self.input.blockSignals(False)
        self.suggestions.clear()
        self.status.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutocompleteApp(Autocompleter())
    window.show()
    sys.exit(app.exec())
