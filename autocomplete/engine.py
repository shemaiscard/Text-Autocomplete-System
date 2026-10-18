"""What the user interface talks to: one call per keystroke, one per pick."""
import gc

from . import fuzzy
from .data import DEFAULT_FREQUENCIES, DEFAULT_WORDS, load_vocabulary
from .learning import LearnedStore
from .trie import DEFAULT_K, RankedTrie


class Autocompleter:
    def __init__(self, words_path=DEFAULT_WORDS, freq_path=DEFAULT_FREQUENCIES,
                 store=None, k=DEFAULT_K, words=None, weights=None, merge=True):
        if words is None or weights is None:
            words, weights = load_vocabulary(words_path, freq_path, merge)
        self.store = store if store is not None else LearnedStore()
        self.store.apply(weights)
        self.k = k
        self.trie = RankedTrie(words, weights, k=k)
        gc.freeze()  # the trie lives for the whole session; keep it out of GC scans

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
