from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QListWidget, QLabel
from PyQt6.QtCore import Qt

class TrieNode:
    def __init__(self):
        self.children = {}  # Each node stores its children in a dictionary
        self.is_end_of_word = False  # Marks if the node represents the end of a word

class Trie:
    def __init__(self):
        self.root = TrieNode()  # Root node of the trie

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()  # Create new node if char doesn't exist
            node = node.children[char]  # Move to the next node
        node.is_end_of_word = True  # Mark the last node as end of word

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not in trie, return empty list
            node = node.children[char]  # Move to the next node
        return self._collect_words(node, prefix)  # Collect all words starting with prefix

    def _collect_words(self, node, prefix):
        words = []
        if node.is_end_of_word:
            words.append(prefix)  # Add prefix if it's a complete word
        for char, next_node in node.children.items():
            words.extend(self._collect_words(next_node, prefix + char))  # Recursively collect words
        return words

class AutocompleteApp(QWidget):
    def __init__(self, trie):
        super().__init__()
        self.trie = trie  # Store the trie for autocompletion
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Autocomplete System")
        self.setGeometry(100, 100, 400, 300)  # Set window size and position
        
        layout = QVBoxLayout()  # Use vertical layout
        
        self.label = QLabel("Start typing to get suggestions:")
        layout.addWidget(self.label)

        self.input = QLineEdit()
        self.input.textChanged.connect(self.on_text_changed)  # Connect text change to handler
        layout.addWidget(self.input)

        self.suggestions = QListWidget()
        self.suggestions.itemClicked.connect(self.on_suggestion_clicked)  # Connect item click to handler
        layout.addWidget(self.suggestions)

        self.setLayout(layout)

    def on_text_changed(self):
        text = self.input.text()
        self.suggestions.clear()  # Clear previous suggestions
        if text:
            last_word = text.split()[-1]  # Get the last word being typed
            matches = self.trie.autocomplete(last_word)  # Get autocomplete suggestions
            self.suggestions.addItems(matches)  # Add suggestions to list

    def on_suggestion_clicked(self, item):
        text = self.input.text()
        words = text.split()
        words[-1] = item.text()  # Replace last word with clicked suggestion
        new_text = ' '.join(words)
        self.input.setText(new_text)  # Update input text

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    trie = Trie()
    with open('words.txt', 'r') as file:
        for line in file:
            trie.insert(line.strip())  # Insert each word from file into trie

    autocomplete_app = AutocompleteApp(trie)
    autocomplete_app.show()
    sys.exit(app.exec())  # Start the application event loop