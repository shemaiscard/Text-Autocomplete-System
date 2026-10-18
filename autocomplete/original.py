"""The trie of the first version of the app, kept unchanged as a baseline.

autocomplete() returns every word under the prefix, in insertion order,
with no ranking. The benchmarks compare the new structures against it.
"""


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
