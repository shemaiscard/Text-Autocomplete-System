"""Prepare the shared benchmark inputs (Python and Rust read the same files).

    python bench/prepare.py --book pg1661.txt --misspellings list.wiki --out bench/cache

Writes, into --out:
  vocab_<n>.tsv       word<TAB>weight, byte-sorted; n = 50000, 100000, 200000
                      (random subsets of words.txt), 466550 (all of words.txt)
                      and the merged vocabulary the app uses (words.txt plus
                      the words of data/word_frequencies_en.tsv)
  prefixes_<n>.json   the typed prefixes per length and the worst prefix
  book_tokens.json    the Gutenberg text, tokenized and split in halves
  picks.txt           in-vocabulary words of the book, in order (learning benchmark)
  misspellings.tsv    misspelling<TAB>correction[,correction...]
  fuzzy_queries.txt   200 misspellings sampled for the latency benchmark
  manifest.json       counts, seeds and SHA-256 of the inputs
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from autocomplete.data import load_vocabulary  # noqa: E402

SUBSET_SIZES = (50_000, 100_000, 200_000)
SEED = 42
TYPED_PER_LENGTH = 20
MAX_LEN = 6
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_vocab(path, weights):
    items = sorted(weights.items(), key=lambda x: x[0].encode("utf-8"))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for w, c in items:
            f.write("%s\t%d\n" % (w, c))


def typed_prefixes(weights):
    """26 letters, then the TYPED_PER_LENGTH prefixes of each length with the
    largest frequency mass (sum of the weights of the words they start)."""
    out = {"1": list("abcdefghijklmnopqrstuvwxyz")}
    for L in range(2, MAX_LEN + 1):
        mass = {}
        for w, c in weights.items():
            p = w[:L]
            if len(p) == L and p.isascii() and p.isalpha() and p.islower():
                mass[p] = mass.get(p, 0) + c
        top = sorted(mass.items(), key=lambda x: (-x[1], x[0]))[:TYPED_PER_LENGTH]
        out[str(L)] = [p for p, _ in top]
    return out


def prefix_file(words, typed):
    counts = {}
    for w in words:
        for L in range(1, min(MAX_LEN, len(w)) + 1):
            p = w[:L]
            counts[p] = counts.get(p, 0) + 1
    worst = {}
    for L in range(1, MAX_LEN + 1):
        cands = [(c, p) for p, c in counts.items() if len(p) == L]
        c, p = max(cands, key=lambda x: (x[0], [-ord(ch) for ch in x[1]]))
        worst[str(L)] = {"prefix": p, "count": c}
    t = {L: [{"prefix": p, "count": counts.get(p, 0)} for p in ps if counts.get(p, 0) > 0]
         for L, ps in typed.items()}
    return {"typed": t, "worst": worst}


def strip_gutenberg(text):
    start = re.search(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M)
    end = re.search(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M)
    if not start or not end:
        raise SystemExit("Gutenberg markers not found")
    return text[start.end():end.start()]


def parse_misspellings(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^([^\s<>=]+)->([^<>=]+)$", line.strip())
            if m:
                corr = [c.strip() for c in m.group(2).split(",") if c.strip()]
                pairs.append((m.group(1), corr))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--misspellings", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "cache"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {"seed": SEED}

    words, weights = load_vocabulary(merge=False)
    manifest["words_txt"] = {"entries": len(words), "with_frequency": sum(1 for v in weights.values() if v > 1)}
    sizes = {}
    full_n = len(weights)
    write_vocab(os.path.join(args.out, "vocab_%d.tsv" % full_n), weights)
    sizes[full_n] = sorted(weights)
    rng = random.Random(SEED)
    for n in SUBSET_SIZES:
        sub = rng.sample(words, n)
        write_vocab(os.path.join(args.out, "vocab_%d.tsv" % n), {w: weights[w] for w in sub})
        sizes[n] = sub
    mwords, mweights = load_vocabulary(merge=True)
    write_vocab(os.path.join(args.out, "vocab_%d.tsv" % len(mweights)), mweights)
    sizes[len(mweights)] = sorted(mweights)
    manifest["merged"] = {"n": len(mweights), "added_from_frequency_file": len(mweights) - full_n,
                          "with_frequency": sum(1 for v in mweights.values() if v > 1)}

    typed = typed_prefixes(weights)
    for n, ws in sizes.items():
        with open(os.path.join(args.out, "prefixes_%d.json" % n), "w", encoding="utf-8") as f:
            json.dump(prefix_file(ws, typed), f, indent=1)
    manifest["sizes"] = sorted(sizes)

    with open(args.book, encoding="utf-8-sig") as f:
        body = strip_gutenberg(f.read())
    tokens = TOKEN.findall(body.lower())
    half = len(tokens) // 2
    with open(os.path.join(args.out, "book_tokens.json"), "w", encoding="utf-8") as f:
        json.dump({"first": tokens[:half], "second": tokens[half:]}, f)
    in_vocab = [t for t in tokens if t in weights]
    in_merged = [t for t in tokens if t in mweights]
    with open(os.path.join(args.out, "picks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(in_merged[:20_000]) + "\n")
    manifest["book"] = {"file": os.path.basename(args.book), "sha256": sha256(args.book),
                        "tokens": len(tokens), "first_half": half, "second_half": len(tokens) - half,
                        "distinct": len(set(tokens)),
                        "tokens_in_words_txt": len(in_vocab), "tokens_in_merged": len(in_merged),
                        "picks_written": min(20_000, len(in_merged))}

    pairs = parse_misspellings(args.misspellings)
    kept = [(m, [c for c in cs if c in mweights]) for m, cs in pairs]
    kept = [(m, cs) for m, cs in kept if cs and m.isascii() and " " not in m]
    with open(os.path.join(args.out, "misspellings.tsv"), "w", encoding="utf-8") as f:
        for m, cs in kept:
            f.write("%s\t%s\n" % (m, ",".join(cs)))
    sample = random.Random(SEED).sample(kept, 200)
    with open(os.path.join(args.out, "fuzzy_queries.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(m for m, _ in sample) + "\n")
    manifest["misspellings"] = {"file": os.path.basename(args.misspellings),
                                "sha256": sha256(args.misspellings), "pairs": len(pairs),
                                "kept_correction_in_vocab": len(kept),
                                "misspelling_in_words_txt": sum(1 for m, _ in kept if m in weights),
                                "misspelling_in_merged": sum(1 for m, _ in kept if m in mweights),
                                "latency_sample": len(sample)}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
