"""Unit tests (standard library only). Run: python -m unittest discover -s tests"""
import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocomplete import Autocompleter, LearnedStore, RankedTrie, SortedIndex  # noqa: E402
from autocomplete import fuzzy  # noqa: E402
from autocomplete.data import base_weights, load_vocabulary  # noqa: E402
from autocomplete.original import Trie as OriginalTrie  # noqa: E402


def brute_topk(weights, prefix, k):
    found = [w for w in weights if w.startswith(prefix)]
    found.sort(key=lambda w: (-weights[w], w))
    return found[:k]


def levenshtein(a, b, transpositions=False):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if (transpositions and i > 1 and j > 1 and a[i - 1] == b[j - 2]
                    and a[i - 2] == b[j - 1]):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def random_vocab(rng, n, alphabet="abcde", max_len=7, weight_values=(1, 2, 3, 5, 8, 50)):
    words = set()
    while len(words) < n:
        words.add("".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_len))))
    return {w: rng.choice(weight_values) for w in sorted(words)}


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.weights = {"the": 900, "then": 300, "there": 300, "they": 500, "theory": 20,
                        "a": 1000, "an": 400, "and": 950, "ant": 5, "zebra": 1}
        self.trie = RankedTrie(list(self.weights), dict(self.weights), k=3)

    def test_precomputed_order(self):
        self.assertEqual(self.trie.suggest("th"), ["the", "they", "then"])
        self.assertEqual(self.trie.suggest("an"), ["and", "an", "ant"])
        self.assertEqual(self.trie.suggest("q"), [])
        self.assertEqual(self.trie.suggest("zebra"), ["zebra"])

    def test_ties_break_alphabetically(self):
        self.assertEqual(self.trie.suggest("then", 3), ["then"])
        self.assertEqual(self.trie.suggest_bestfirst("the", 5),
                         ["the", "they", "then", "there", "theory"])

    def test_methods_agree_on_random_vocabularies(self):
        rng = random.Random(7)
        for _ in range(20):
            w = random_vocab(rng, 300)
            for k in (1, 3, 10):
                trie = RankedTrie(list(w), dict(w), k=k)
                prefixes = {"", "a", "b", "ab", "abc", "eee", "q"} | {x[:2] for x in w}
                for p in prefixes:
                    expect = brute_topk(w, p, k)
                    self.assertEqual(trie.suggest(p, k), expect, (p, k))
                    self.assertEqual(trie.suggest_bestfirst(p, k), expect, (p, k))
                    self.assertEqual(trie.suggest_heap(p, k), expect, (p, k))

    def test_larger_k_than_cache_falls_back(self):
        self.assertEqual(self.trie.suggest("", 10), brute_topk(self.weights, "", 10))

    def test_sorted_index(self):
        idx = SortedIndex(list(self.weights), self.weights)
        self.assertEqual(idx.suggest("th", 3), ["the", "then", "theory"])
        self.assertEqual(idx.suggest_ranked("th", 3), ["the", "they", "then"])
        self.assertEqual(idx.range("x"), (idx.range("x")[0],) * 2)

    def test_original_trie_returns_everything(self):
        t = OriginalTrie()
        for w in self.weights:
            t.insert(w)
        self.assertEqual(sorted(t.autocomplete("the")), ["the", "then", "theory", "there", "they"])
        self.assertEqual(t.autocomplete("x"), [])

    def test_base_weights_floor_and_case(self):
        w = base_weights(["The", "cat", "zzzq"], {"the": 50, "cat": 7})
        self.assertEqual(w, {"The": 50, "cat": 7, "zzzq": 1})
        w = base_weights(["HE", "he", "Holmes"], {"he": 900, "holmes": 40})
        self.assertEqual(w, {"HE": 1, "he": 900, "Holmes": 40})

    def test_merged_vocabulary(self):
        with tempfile.TemporaryDirectory() as d:
            wp, fp = os.path.join(d, "w.txt"), os.path.join(d, "f.tsv")
            with open(wp, "w", encoding="utf-8") as f:
                f.write("OF\ncat\n")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("# comment\nof\t900\ncat\t7\nweb\t3\n")
            words, weights = load_vocabulary(wp, fp, merge=False)
            self.assertEqual((words, weights), (["OF", "cat"], {"OF": 900, "cat": 7}))
            words, weights = load_vocabulary(wp, fp)
            self.assertEqual(words, ["OF", "cat", "of", "web"])
            self.assertEqual(weights, {"OF": 1, "cat": 7, "of": 900, "web": 3})


class LearningTests(unittest.TestCase):
    def test_learn_matches_full_rebuild(self):
        rng = random.Random(11)
        for trial in range(15):
            w = random_vocab(rng, 250)
            k = rng.choice((1, 3, 5, 10))
            live = RankedTrie(list(w), dict(w), k=k)
            words = list(w)
            for _ in range(40):
                pick = rng.choice(words)
                live.learn(pick, rng.choice((1, 4, 100)))
            fresh = RankedTrie(list(w), dict(live.weight), k=k)
            stack = [(live.root, fresh.root)]
            while stack:
                a, b = stack.pop()
                self.assertEqual(a.top, b.top)
                self.assertEqual(a.best, b.best)
                for ch, child in (a.children or {}).items():
                    stack.append((child, b.children[ch]))

    def test_learned_word_rises(self):
        w = {"car": 100, "card": 90, "care": 80, "carp": 1}
        t = RankedTrie(list(w), dict(w), k=2)
        self.assertNotIn("carp", t.suggest("ca"))
        t.learn("carp", 150)
        self.assertEqual(t.suggest("ca"), ["carp", "car"])
        self.assertEqual(t.suggest_bestfirst("ca", 2), ["carp", "car"])

    def test_learn_rejects_unknown_words(self):
        t = RankedTrie(["abc"], {"abc": 1}, k=2)
        with self.assertRaises(KeyError):
            t.learn("ab", 5)
        with self.assertRaises(KeyError):
            t.learn("abcd", 5)
        with self.assertRaises(ValueError):
            t.learn("abc", 0)

    def test_store_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "learned.json")
            s = LearnedStore(path, delta=10)
            s.record("carp")
            s.record("carp")
            s.save()
            again = LearnedStore(path, delta=10)
            self.assertEqual(again.picks, {"carp": 2})
            weights = {"carp": 1, "car": 5}
            again.apply(weights)
            self.assertEqual(weights, {"carp": 21, "car": 5})
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["version"], 1)

    def test_store_ignores_corrupt_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "learned.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertEqual(LearnedStore(path).picks, {})

    def test_autocompleter_learns_and_persists(self):
        w = {"car": 100, "card": 90, "care": 80, "carp": 1}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "learned.json")
            ac = Autocompleter(store=LearnedStore(path, delta=1000), k=2,
                               words=list(w), weights=dict(w))
            self.assertEqual(ac.suggest("ca"), (["car", "card"], False))
            self.assertTrue(ac.accept("carp"))
            self.assertFalse(ac.accept("nope"))
            self.assertEqual(ac.suggest("ca")[0][0], "carp")
            reopened = Autocompleter(store=LearnedStore(path, delta=1000), k=2,
                                     words=list(w), weights=dict(w))
            self.assertEqual(reopened.suggest("ca")[0][0], "carp")


class FuzzyTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = random.Random(3)
        w = random_vocab(rng, 400, alphabet="abcd", max_len=6)
        trie = RankedTrie(list(w), dict(w), k=5)
        for _ in range(60):
            q = "".join(rng.choice("abcde") for _ in range(rng.randint(1, 7)))
            for d in (1, 2):
                for tr in (False, True):
                    got = sorted(fuzzy.search(trie, q, d, tr))
                    expect = sorted((levenshtein(q, x, tr), x) for x in w
                                    if levenshtein(q, x, tr) <= d)
                    self.assertEqual(got, expect, (q, d, tr))

    def test_suggest_prefers_distance_then_weight(self):
        w = {"search": 1000, "perch": 10, "starch": 500, "serge": 2}
        trie = RankedTrie(list(w), dict(w), k=3)
        self.assertEqual(fuzzy.suggest(trie, "serch", 3), ["search", "perch", "starch"])
        self.assertEqual(fuzzy.suggest(trie, "teh", 3, transpositions=False), [])
        w2 = {"the": 10, "ten": 1}
        t2 = RankedTrie(list(w2), dict(w2), k=2)
        self.assertEqual(fuzzy.suggest(t2, "teh", 1), ["ten"])
        self.assertEqual(fuzzy.suggest(t2, "teh", 1, transpositions=True), ["the"])

    def test_autocompleter_falls_back_to_fuzzy(self):
        w = {"search": 1000, "season": 10}
        with tempfile.TemporaryDirectory() as d:
            ac = Autocompleter(store=LearnedStore(os.path.join(d, "l.json")), k=3,
                               words=list(w), weights=dict(w))
            self.assertEqual(ac.suggest("serch"), (["search"], True))
            self.assertEqual(ac.suggest("sea"), (["search", "season"], False))
            self.assertEqual(ac.suggest(""), ([], False))


if __name__ == "__main__":
    unittest.main()
