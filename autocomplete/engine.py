"""What the user interface talks to: one call per keystroke, one per pick."""
from . import fuzzy
from .data import DEFAULT_FREQUENCIES, DEFAULT_WORDS, base_weights, load_frequencies, load_words
from .learning import LearnedStore
from .trie import DEFAULT_K, RankedTrie


class Autocompleter:
    def __init__(self, words_path=DEFAULT_WORDS, freq_path=DEFAULT_FREQUENCIES,
                 store=None, k=DEFAULT_K, words=None, weights=None):
        if words is None:
            words = load_words(words_path)
        if weights is None:
            weights = base_weights(words, load_frequencies(freq_path))
        self.store = store if store is not None else LearnedStore()
        self.store.apply(weights)
        self.k = k
        self.trie = RankedTrie(words, weights, k=k)

    def suggest(self, prefix):
        """Return (words, is_fuzzy). Fuzzy matches only when nothing completes the prefix."""
        if not prefix:
            return [], False
        found = self.trie.suggest(prefix, self.k)
        if found:
            return found, False
        return fuzzy.suggest(self.trie, prefix, self.k), True

    def accept(self, word, save=True):
        """The user picked a suggestion: learn it and persist the counts."""
        if word not in self.trie:
            return False
        self.trie.learn(word, self.store.delta)
        self.store.record(word)
        if save:
            self.store.save()
        return True
