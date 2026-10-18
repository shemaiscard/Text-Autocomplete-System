"""Build data/word_frequencies_en.tsv from the wordfreq package.

The file gives English frequencies in occurrences per billion words
(wordfreq "large" list, version 3.1.1) for
  * every lowercase form of a word in words.txt that wordfreq knows, and
  * every other wordfreq word made of the letters a-z (with inner
    apostrophes), which the app adds to its vocabulary.
Words of words.txt that are missing get the floor weight at load time
(see autocomplete/data.py).

This script is only needed to regenerate the file:

    python -m pip install --target ./.wordfreq wordfreq==3.1.1
    PYTHONPATH=./.wordfreq python tools/build_frequencies.py

The data is derived from wordfreq and is licensed CC BY-SA 4.0; see
data/FREQUENCIES_LICENSE.txt for the attribution it requires.
"""
import os
import re
import sys

import wordfreq
from importlib.metadata import version as _pkg_version

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    words_path = os.path.join(ROOT, "words.txt")
    out_path = os.path.join(ROOT, "data", "word_frequencies_en.tsv")
    freq = wordfreq.get_frequency_dict("en", "large")
    forms = set()
    with open(words_path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                forms.add(w.lower())
    token = re.compile(r"[a-z]+(?:'[a-z]+)*")
    rows = []
    for w, p in freq.items():
        if p and (w in forms or token.fullmatch(w)):
            rows.append((w, max(1, round(p * 1e9))))
    rows.sort(key=lambda r: (-r[1], r[0]))
    version = _pkg_version("wordfreq")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# English word frequencies, occurrences per billion words.\n")
        f.write("# Derived from wordfreq %s (Robyn Speer), 'large' English list,\n" % version)
        f.write("# Lowercase forms of words.txt plus wordfreq words of letters a-z.\n")
        f.write("# License: CC BY-SA 4.0. See data/FREQUENCIES_LICENSE.txt.\n")
        for w, c in rows:
            f.write("%s\t%d\n" % (w, c))
    print("wrote %d rows to %s" % (len(rows), out_path), file=sys.stderr)


if __name__ == "__main__":
    main()
