"""Keystroke savings evaluation on the second half of Sherlock Holmes."""
import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from autocomplete.learning import LEARN_DELTA
from autocomplete.sorted_index import SortedIndex
from autocomplete.trie import RankedTrie
from bench.common import load_vocab


def count_keystrokes_for_word(word, suggest_fn, ks=(1, 5, 10)):
    """Simulate typing `word` letter-by-letter.

    After typing p letters (p = 1..len(word)), ask for up to max(ks) suggestions.
    When `word` appears at 1-based rank `r`:
      for each k in ks where r <= k (and not yet completed):
        keystrokes_used = min(len(word), p + 1)
        letters_saved = len(word) - keystrokes_used

    If word never appears in top-k:
      keystrokes_used = len(word)
      letters_saved = 0

    Returns {k: {"keystrokes": int, "saved": int}}
    """
    L = len(word)
    max_k = max(ks)
    res = {k: {"keystrokes": L, "saved": 0} for k in ks}
    if L <= 1:
        return res

    pending = set(ks)
    for p in range(1, L + 1):
        prefix = word[:p]
        top = suggest_fn(prefix, max_k)
        if word in top:
            rank = top.index(word) + 1
            completed_now = []
            for k in pending:
                if rank <= k:
                    cost = min(L, p + 1)
                    res[k] = {"keystrokes": cost, "saved": L - cost}
                    completed_now.append(k)
            for k in completed_now:
                pending.remove(k)
            if not pending:
                break
    return res


def evaluate_tokens(suggest_fn, tokens, vocab_set, ks=(1, 5, 10)):
    """Evaluate keystroke savings over all tokens in a corpus."""
    total_tokens = len(tokens)
    total_letters = sum(len(t) for t in tokens)

    in_vocab_tokens = [t for t in tokens if t in vocab_set]
    in_vocab_token_count = len(in_vocab_tokens)
    in_vocab_letters = sum(len(t) for t in in_vocab_tokens)

    word_counts = Counter(in_vocab_tokens)
    word_stats = {}
    for w in word_counts:
        word_stats[w] = count_keystrokes_for_word(w, suggest_fn, ks)

    out = {}
    for k in ks:
        saved = sum(word_stats[w][k]["saved"] * count for w, count in word_counts.items())
        used = total_letters - saved
        out[str(k)] = {
            "letters_saved": saved,
            "keystrokes_used": used,
            "ksr": round(saved / total_letters, 4) if total_letters else 0.0,
            "ksr_percent": round(100.0 * saved / total_letters, 2) if total_letters else 0.0,
            "ksr_in_vocab": round(saved / in_vocab_letters, 4) if in_vocab_letters else 0.0,
            "ksr_in_vocab_percent": round(100.0 * saved / in_vocab_letters, 2) if in_vocab_letters else 0.0,
        }
    return {
        "total_tokens": total_tokens,
        "total_letters": total_letters,
        "in_vocab_tokens": in_vocab_token_count,
        "in_vocab_letters": in_vocab_letters,
        "oov_tokens": total_tokens - in_vocab_token_count,
        "oov_letters": total_letters - in_vocab_letters,
        "by_k": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Load test tokens (second half of Sherlock Holmes)
    with open(os.path.join(a.cache, "book_tokens.json"), encoding="utf-8") as f:
        tokens = json.load(f)["second"]

    # Load picks for learning condition
    with open(os.path.join(a.cache, "picks.txt"), encoding="utf-8") as f:
        picks = [w for w in f.read().splitlines() if w]

    print("Tokens in second half:", len(tokens), "Total letters:", sum(len(t) for t in tokens))

    results = {
        "total_tokens": len(tokens),
        "total_letters": sum(len(t) for t in tokens),
        "conditions": {},
    }

    # 1-3. Merged 677,224 vocabulary
    print("Loading merged vocabulary (677,224 words)...")
    m_words, m_weights = load_vocab(os.path.join(a.cache, "vocab_677224.tsv"))
    m_vocab_set = set(m_words)

    print("Condition 1: Alphabetical top-k (merged)...")
    s_idx = SortedIndex(m_words, m_weights)
    results["conditions"]["alphabetical_merged"] = evaluate_tokens(
        s_idx.suggest, tokens, m_vocab_set
    )

    print("Condition 2: Ranked top-k (merged)...")
    trie = RankedTrie(m_words, m_weights, k=10)
    results["conditions"]["ranked_merged"] = evaluate_tokens(
        trie.suggest, tokens, m_vocab_set
    )

    print("Condition 3: Ranked plus learning (merged)...")
    learned_trie = RankedTrie(m_words, dict(m_weights), k=10)
    for p in picks:
        if p in m_weights:
            learned_trie.learn(p, LEARN_DELTA)
    results["conditions"]["ranked_learned_merged"] = evaluate_tokens(
        learned_trie.suggest, tokens, m_vocab_set
    )
    del s_idx, trie, learned_trie, m_words, m_weights, m_vocab_set

    # 4. Conditions 1-3 with words.txt only (466,550 words)
    print("Loading words.txt vocabulary (466,550 words)...")
    w_words, w_weights = load_vocab(os.path.join(a.cache, "vocab_466550.tsv"))
    w_vocab_set = set(w_words)

    print("Condition 4a: Alphabetical top-k (words.txt only)...")
    s_idx_w = SortedIndex(w_words, w_weights)
    results["conditions"]["alphabetical_wordstxt"] = evaluate_tokens(
        s_idx_w.suggest, tokens, w_vocab_set
    )

    print("Condition 4b: Ranked top-k (words.txt only)...")
    trie_w = RankedTrie(w_words, w_weights, k=10)
    results["conditions"]["ranked_wordstxt"] = evaluate_tokens(
        trie_w.suggest, tokens, w_vocab_set
    )

    print("Condition 4c: Ranked plus learning (words.txt only)...")
    learned_trie_w = RankedTrie(w_words, dict(w_weights), k=10)
    for p in picks:
        if p in w_weights:
            learned_trie_w.learn(p, LEARN_DELTA)
    results["conditions"]["ranked_learned_wordstxt"] = evaluate_tokens(
        learned_trie_w.suggest, tokens, w_vocab_set
    )

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved keystroke savings evaluation to", a.out)


if __name__ == "__main__":
    main()
