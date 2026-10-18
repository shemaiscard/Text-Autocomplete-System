"""Python benchmark: one process per (vocabulary, structure) so RSS deltas are clean.

    python bench/bench_python.py --cache bench/cache --n 466550 --structure ranked --out r.json

structure: original  the first version's trie, returns every completion
           ranked    RankedTrie: precomputed top-k, best-first, query-time heap,
                     plus learning, JSON boundary and fuzzy (with --full)
           sorted    sorted array: alphabetical first-k and ranked range scan
"""
import argparse
import gc
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.setrecursionlimit(10_000)

from common import environment, load_vocab, pc, peak_mb, rss_mb, sample_calls, summarize  # noqa: E402

from autocomplete import fuzzy  # noqa: E402
from autocomplete.learning import LEARN_DELTA, LearnedStore  # noqa: E402
from autocomplete.original import Trie as OriginalTrie  # noqa: E402
from autocomplete.sorted_index import SortedIndex  # noqa: E402
from autocomplete.trie import RankedTrie  # noqa: E402


def time_method(f, k, prefixes, samples):
    """prefixes: {L: [{"prefix", "count"}]} -> per-length pooled stats + per-prefix detail."""
    out = {}
    for L, items in prefixes.items():
        pooled, per = [], []
        for it in items:
            p = it["prefix"]
            xs, reps = sample_calls(f, (p, k), samples)
            pooled += xs
            per.append({"prefix": p, "count": it["count"], "reps": reps,
                        "median_us": summarize(xs)["median"], "samples_us": xs})
        out[L] = {"summary": summarize(pooled), "per_prefix": per}
    return out


def worst_cases(f, k, worst, samples):
    out = {}
    for L, it in worst.items():
        xs, reps = sample_calls(f, (it["prefix"], k), samples)
        out[L] = {"prefix": it["prefix"], "count": it["count"], "reps": reps,
                  "summary": summarize(xs), "samples_us": xs}
    return out


def noop(a, b):
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--structure", choices=["original", "ranked", "sorted"], required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--full", action="store_true", help="also learning, boundary and fuzzy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rec = {"language": "python", "structure": a.structure, "k": a.k, "samples": a.samples,
           "env": environment(REPO)}
    with open(os.path.join(a.cache, "prefixes_%d.json" % a.n), encoding="utf-8") as f:
        pref = json.load(f)
    mem = {"start": rss_mb()}
    words, weights = load_vocab(os.path.join(a.cache, "vocab_%d.tsv" % a.n))
    mem["after_vocab"] = rss_mb()
    rec["n_words"] = len(words)
    xs, _ = sample_calls(noop, ("a", 1), 15)
    rec["empty_call_us"] = summarize(xs)
    methods = {}
    gc.collect()

    if a.structure == "original":
        t0 = pc()
        t = OriginalTrie()
        for w in words:
            t.insert(w)
        rec["build_s"] = (pc() - t0) / 1e9
        mem["after_structure"] = rss_mb()
        gc.freeze()
        calls = {"original": lambda p, k: t.autocomplete(p)}
    elif a.structure == "ranked":
        t0 = pc()
        t = RankedTrie(words, weights, k=a.k, cache_topk=False, cache_best=False)
        rec["build_s"] = (pc() - t0) / 1e9
        rec["n_nodes"] = t.node_count
        mem["after_structure"] = rss_mb()
        t0 = pc()
        t.build_topk()
        rec["precompute_topk_s"] = (pc() - t0) / 1e9
        mem["after_topk"] = rss_mb()
        t0 = pc()
        t.build_best()
        rec["precompute_best_s"] = (pc() - t0) / 1e9
        mem["after_best"] = rss_mb()
        gc.freeze()
        calls = {"topk": t.suggest, "bestfirst": t.suggest_bestfirst, "heap": t.suggest_heap}
    else:
        t0 = pc()
        idx = SortedIndex(words, weights)
        rec["build_s"] = (pc() - t0) / 1e9
        mem["after_structure"] = rss_mb()
        gc.freeze()
        calls = {"sorted_alpha": idx.suggest, "sorted_ranked": idx.suggest_ranked}

    for name, f in calls.items():
        gc.collect()
        print("  timing", name, file=sys.stderr)
        methods[name] = {"by_length": time_method(f, a.k, pref["typed"], a.samples),
                         "worst": worst_cases(f, a.k, pref["worst"], a.samples)}
    rec["methods"] = methods

    if a.structure == "ranked":
        rec["topk_results"] = {it["prefix"]: t.suggest(it["prefix"], a.k)
                               for items in pref["typed"].values() for it in items}

    if a.full and a.structure in ("ranked", "original"):
        # JSON boundary proxy: what a web or IPC front end would have to ship.
        def boundary(prefix, k):
            if a.structure == "ranked":
                res = [{"word": w, "score": weights[w]} for w in t.suggest(prefix, k)]
            else:
                res = t.autocomplete(prefix)
            return json.loads(json.dumps(res))

        def encode_only(prefix, k):
            if a.structure == "ranked":
                res = [{"word": w, "score": weights[w]} for w in t.suggest(prefix, k)]
            else:
                res = t.autocomplete(prefix)
            return json.dumps(res)

        gc.collect()
        print("  boundary", file=sys.stderr)
        rec["boundary"] = {"lookup_encode_decode": time_method(boundary, a.k, pref["typed"], a.samples),
                           "lookup_encode": time_method(encode_only, a.k, pref["typed"], a.samples),
                           "worst_lookup_encode_decode": worst_cases(boundary, a.k, pref["worst"], a.samples)}
        sizes = {}
        for L, items in pref["typed"].items():
            sizes[L] = [len(encode_only(it["prefix"], a.k)) for it in items]
        rec["boundary"]["payload_bytes"] = {L: summarize(v) for L, v in sizes.items()}

    if a.full and a.structure == "ranked":
        print("  fuzzy", file=sys.stderr)
        # Fuzzy fallback latency on sampled misspellings.
        with open(os.path.join(a.cache, "fuzzy_queries.txt"), encoding="utf-8") as f:
            queries = [q for q in f.read().split("\n") if q]
        lat, used_d2, results = [], 0, {}
        for q in queries:
            if len(fuzzy.search(t, q, 1)) < a.k:
                used_d2 += 1
            xs, _ = sample_calls(lambda s, k: fuzzy.suggest(t, s, k), (q, a.k), 5, target_ns=0)
            lat.append(summarize(xs)["median"])
            results[q] = fuzzy.suggest(t, q, a.k)
        d1, d2 = [], []
        for q in queries:
            d1.append(summarize(sample_calls(lambda s, k: fuzzy.search(t, s, 1), (q, 0), 3, target_ns=0)[0])["median"])
            d2.append(summarize(sample_calls(lambda s, k: fuzzy.search(t, s, 2), (q, 0), 3, target_ns=0)[0])["median"])

        def lev(x, y):
            prev = list(range(len(y) + 1))
            for i in range(1, len(x) + 1):
                cur = [i]
                xi = x[i - 1]
                for j in range(1, len(y) + 1):
                    v = prev[j - 1] + (xi != y[j - 1])
                    if prev[j] + 1 < v:
                        v = prev[j] + 1
                    if cur[j - 1] + 1 < v:
                        v = cur[j - 1] + 1
                    cur.append(v)
                prev = cur
            return prev[-1]

        def brute(q, k):
            return [w for w in words if abs(len(w) - len(q)) <= 2 and lev(q, w) <= 2]

        bf = []
        for q in queries[:20]:
            xs, _ = sample_calls(brute, (q, 0), 3, target_ns=0)
            assert sorted(brute(q, 0)) == sorted(w for _, w in fuzzy.search(t, q, 2))
            bf.append(summarize(xs)["median"])
        rec["fuzzy"] = {"queries": len(queries), "needed_distance_2": used_d2,
                        "suggest_median_us_per_query": lat, "suggest_summary": summarize(lat),
                        "search_d1_summary": summarize(d1), "search_d2_summary": summarize(d2),
                        "bruteforce_d2_first20_summary": summarize(bf),
                        "trie_d2_first20_summary": summarize(d2[:20]), "results": results}

        print("  learn", file=sys.stderr)
        # Adaptive learning: per-pick update cost, then a full rebuild for contrast.
        with open(os.path.join(a.cache, "picks.txt"), encoding="utf-8") as f:
            picks = [w for w in f.read().split("\n") if w and w in weights]
        gc.collect()
        times = []
        for w in picks:
            t0 = pc()
            t.learn(w, LEARN_DELTA)
            times.append((pc() - t0) / 1000.0)
        t0 = pc()
        fresh = RankedTrie(words, dict(t.weight), k=a.k, cache_topk=False, cache_best=False)
        rebuild_build = (pc() - t0) / 1e9
        t0 = pc()
        fresh.build_topk()
        fresh.build_best()
        rebuild_caches = (pc() - t0) / 1e9
        same = all(fresh.suggest(it["prefix"]) == t.suggest(it["prefix"])
                   for items in pref["typed"].values() for it in items)
        del fresh
        with tempfile.TemporaryDirectory() as d:
            store = LearnedStore(os.path.join(d, "learned.json"))
            for w in picks:
                store.record(w)
            save = []
            for _ in range(20):
                t0 = pc()
                store.save()
                save.append((pc() - t0) / 1000.0)
            size = os.path.getsize(store.path)
        rec["learn"] = {"picks": len(picks), "distinct": len(set(picks)), "delta": LEARN_DELTA,
                        "update_us": summarize(times), "update_us_raw": times,
                        "rebuild_trie_s": rebuild_build, "rebuild_caches_s": rebuild_caches,
                        "matches_rebuild_on_typed_prefixes": same,
                        "save_json_us": summarize(save), "json_bytes": size,
                        "storage_note": "saved to a tmpfs folder; no fsync"}

    mem["peak"] = peak_mb()
    rec["mem_mb"] = mem
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    print("done", a.structure, a.n, file=sys.stderr)


if __name__ == "__main__":
    main()
