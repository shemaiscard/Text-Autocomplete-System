"""Adaptive learning: words the user picks rank higher next time.

Each pick adds LEARN_DELTA to the word's weight. The counts are kept in a
small JSON file in the user's data folder:

    Linux    $XDG_DATA_HOME/TextAutocomplete/learned.json (~/.local/share/...)
    macOS    ~/Library/Application Support/TextAutocomplete/learned.json
    Windows  %APPDATA%/TextAutocomplete/learned.json
"""
import json
import os
import sys
import tempfile

# One pick adds 100,000 occurrences per billion words, the frequency of a
# word at Zipf 5 (for example "service" or "simple"). Fixed in advance,
# not tuned on the evaluation text.
LEARN_DELTA = 100_000
APP_DIR = "TextAutocomplete"


def default_data_dir():
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_DIR)


class LearnedStore:
    """Pick counts per word, persisted as JSON."""

    def __init__(self, path=None, delta=LEARN_DELTA):
        self.path = path or os.path.join(default_data_dir(), "learned.json")
        self.delta = delta
        self.picks = {}
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            picks = data.get("picks", {})
            self.picks = {str(w): int(c) for w, c in picks.items() if int(c) > 0}
        except (OSError, ValueError, AttributeError, TypeError):
            self.picks = {}
        return self.picks

    def record(self, word):
        self.picks[word] = self.picks.get(word, 0) + 1

    def apply(self, weights):
        """Add the learned weight to a fresh weight table (before caches are built)."""
        for w, c in self.picks.items():
            if w in weights:
                weights[w] += self.delta * c

    def save(self):
        """Write atomically: a crash never leaves a half-written file."""
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".learned-", suffix=".json", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "delta": self.delta, "picks": self.picks},
                          f, separators=(",", ":"), sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
