# Text Autocomplete System

A desktop word-completion app. Type in the box and, after a short pause, it shows the ten most likely completions of the word you are typing, ranked by how common each word is in English. Click a suggestion to complete the word. Words you pick move up the list next time. If nothing starts with what you typed, the app offers close spellings instead.

## Features

- Suggestions ranked by real English word frequency, across 677,224 words
- Lookups stay under a microsecond, however large the word list
- Learns from your picks and remembers them between sessions
- Spelling suggestions when a word has no completions
- A Rust core with the same engine, for use from other front ends

## Run the app

```
pip install PyQt6
python "Text Autocomplete System.py"
```

## Learned words

The words you pick are saved in a small file, `learned.json`:

| System  | Location |
|---------|----------|
| Linux   | `~/.local/share/TextAutocomplete/learned.json` (or under `$XDG_DATA_HOME`) |
| macOS   | `~/Library/Application Support/TextAutocomplete/learned.json` |
| Windows | `%APPDATA%\TextAutocomplete\learned.json` |

To reset what the app has learned, close it and delete that file.

## Use the engine from Python

```python
from autocomplete.engine import Autocompleter

ac = Autocompleter()
words, is_spelling_suggestion = ac.suggest("progr")
ac.accept("program")   # the user picked it, so it ranks higher next time
```

## Rust core

`rust-core/` contains the same engine as a Rust library. A front end makes one call per pause in typing and gets back only the top words.

```
cd rust-core
cargo test --release
```

## Tests

```
python -m unittest discover -s tests
```

## Data and credits

- `words.txt`: the original English word list.
- `data/word_frequencies_en.tsv`: word frequencies derived from [wordfreq](https://github.com/rspeer/wordfreq) by Robyn Speer, which draws on SUBTLEX, Google Books Ngrams, OpenSubtitles, Wikipedia and other sources. This file is licensed under CC BY-SA 4.0; see `data/FREQUENCIES_LICENSE.txt`.

## License

The code is released under the MIT License (see `LICENSE`). The frequency data keeps its CC BY-SA 4.0 license.
