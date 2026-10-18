"""Loading the word list and the corpus frequencies."""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PACKAGE_DIR)
DEFAULT_WORDS = os.path.join(REPO_DIR, "words.txt")
DEFAULT_FREQUENCIES = os.path.join(REPO_DIR, "data", "word_frequencies_en.tsv")

# Weight of a word that has no corpus frequency. Every listed frequency is
# at least 10 per billion, so unlisted words always rank below listed ones.
FLOOR_WEIGHT = 1


def load_words(path=DEFAULT_WORDS):
    """Return the non-empty lines of a one-word-per-line file, in file order."""
    words = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words


def load_frequencies(path=DEFAULT_FREQUENCIES):
    """Return {lowercase word: occurrences per billion words}."""
    freqs = {}
    if not os.path.exists(path):
        return freqs
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                freqs[parts[0]] = int(parts[1])
    return freqs


def base_weights(words, freqs, floor=FLOOR_WEIGHT):
    """Corpus weight for every word.

    A frequency belongs to the lowercase form. A word with capitals (for
    example "HE" or "Holmes") takes the frequency of its lowercase form only
    when that form is not itself in the vocabulary; otherwise it gets the floor.
    """
    present = set(words)
    get = freqs.get
    out = {}
    for w in words:
        low = w.lower()
        out[w] = get(low, floor) if (low == w or low not in present) else floor
    return out


def load_vocabulary(words_path=DEFAULT_WORDS, freq_path=DEFAULT_FREQUENCIES, merge=True):
    """Return (words, weights).

    merge=True (the app default) adds every word of the frequency file that
    words.txt lacks in exactly that spelling, so common lowercase words such as
    "of", "it" and "his", which words.txt only lists as "OF", "IT" and "HIS",
    become completable. merge=False keeps words.txt alone.
    """
    words = load_words(words_path)
    freqs = load_frequencies(freq_path)
    if merge:
        have = set(words)
        words = words + sorted(w for w in freqs if w not in have)
    return words, base_weights(words, freqs)
