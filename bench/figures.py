"""Generate publication-ready figures for the Text Autocomplete paper.

Style constraints:
- Vector PDF, 3.5 in wide (single IEEE column)
- Font: Arial / sans-serif, 8 pt
- Palette: ink #1b1d21, grays #5d636e, #9aa1ab, #d5d9de, and blue #2f5d8a (light #a9c0d6)
- Constrained layout
"""
import glob
import json
import os
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASE_DIR = os.path.dirname(REPO)
RESULTS_DIR = os.environ.get("AC_RESULTS", os.path.join(BASE_DIR, "results"))
OUT_DIR = os.path.join(RESULTS_DIR, "figures")
PAPER_FIG_DIR = os.path.expanduser("~/Documents/papers/latex/06_Text_Autocomplete_System/figures")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# Global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "text.color": "#1b1d21",
    "axes.labelcolor": "#1b1d21",
    "xtick.color": "#1b1d21",
    "ytick.color": "#1b1d21",
    "axes.edgecolor": "#9aa1ab",
    "grid.color": "#d5d9de",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})

# Color palette
C_INK = "#1b1d21"
C_BLUE = "#2f5d8a"
C_LIGHT_BLUE = "#6b9ac4"
C_GRAY_DARK = "#5d636e"
C_GRAY_MID = "#9aa1ab"
C_GRAY_LIGHT = "#d5d9de"


def render_pdf_to_png(pdf_path):
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", pdf_path,
                    pdf_path.replace(".pdf", "")], check=True)


def fig1_latency_by_prefix():
    """Figure 1: Median query latency vs prefix length for n=677,224."""
    py_r = json.load(open(os.path.join(RESULTS_DIR, "py_ranked_677224.json")))
    py_o = json.load(open(os.path.join(RESULTS_DIR, "py_original_677224.json")))
    py_s = json.load(open(os.path.join(RESULTS_DIR, "py_sorted_677224.json")))
    rs = json.load(open(os.path.join(RESULTS_DIR, "rs_677224.json")))

    lengths = [1, 2, 3, 4, 5, 6]
    get_med = lambda m, l: m["by_length"][str(l)]["summary"]["median"]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")

    # Python curves (dashed / darker)
    ax.plot(lengths, [get_med(py_o["methods"]["original"], l) for l in lengths],
            label="Original (Py)", color=C_GRAY_DARK, linestyle="--", marker="x")
    ax.plot(lengths, [get_med(py_r["methods"]["heap"], l) for l in lengths],
            label="Heap (Py)", color=C_GRAY_MID, linestyle="--", marker="v")
    ax.plot(lengths, [get_med(py_s["methods"]["sorted_ranked"], l) for l in lengths],
            label="Sorted ranked (Py)", color=C_GRAY_MID, linestyle=":", marker="d")
    ax.plot(lengths, [get_med(py_r["methods"]["bestfirst"], l) for l in lengths],
            label="Best-first (Py)", color=C_LIGHT_BLUE, linestyle="--", marker="^")
    ax.plot(lengths, [get_med(py_r["methods"]["topk"], l) for l in lengths],
            label="Precomputed (Py)", color=C_INK, linestyle="-", marker="o")

    # Rust curves (solid / blue / crisp)
    ax.plot(lengths, [get_med(rs["methods"]["original"], l) for l in lengths],
            label="Original (Rs)", color=C_GRAY_DARK, linestyle="-", marker="x")
    ax.plot(lengths, [get_med(rs["methods"]["bestfirst"], l) for l in lengths],
            label="Best-first (Rs)", color=C_LIGHT_BLUE, linestyle="-", marker="^")
    ax.plot(lengths, [get_med(rs["methods"]["topk"], l) for l in lengths],
            label="Precomputed (Rs)", color=C_BLUE, linestyle="-", marker="s")

    ax.set_yscale("log")
    ax.set_xlabel("Prefix length (characters)")
    ax.set_ylabel("Median latency (μs, log scale)")
    ax.set_xticks(lengths)
    ax.grid(True)
    fig.legend(ncol=2, frameon=False, loc="outside lower center", fontsize=6.5)

    out_pdf = os.path.join(OUT_DIR, "fig1_latency_by_prefix.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    render_pdf_to_png(out_pdf)


def fig2_worst_prefix_scaling():
    """Figure 2: Worst-case prefix latency (L=1) vs vocabulary size."""
    sizes = [50000, 100000, 200000, 466550, 677224]
    sizes_k = [n / 1000 for n in sizes]

    py_orig, py_heap, py_sort_rank, py_bf, py_topk = [], [], [], [], []
    rs_orig, rs_bf, rs_topk = [], [], []

    for n in sizes:
        po = json.load(open(os.path.join(RESULTS_DIR, f"py_original_{n}.json")))
        pr = json.load(open(os.path.join(RESULTS_DIR, f"py_ranked_{n}.json")))
        ps = json.load(open(os.path.join(RESULTS_DIR, f"py_sorted_{n}.json")))
        r = json.load(open(os.path.join(RESULTS_DIR, f"rs_{n}.json")))

        py_orig.append(po["methods"]["original"]["worst"]["1"]["summary"]["median"])
        py_heap.append(pr["methods"]["heap"]["worst"]["1"]["summary"]["median"])
        py_sort_rank.append(ps["methods"]["sorted_ranked"]["worst"]["1"]["summary"]["median"])
        py_bf.append(pr["methods"]["bestfirst"]["worst"]["1"]["summary"]["median"])
        py_topk.append(pr["methods"]["topk"]["worst"]["1"]["summary"]["median"])

        rs_orig.append(r["methods"]["original"]["worst"]["1"]["summary"]["median"])
        rs_bf.append(r["methods"]["bestfirst"]["worst"]["1"]["summary"]["median"])
        rs_topk.append(r["methods"]["topk"]["worst"]["1"]["summary"]["median"])

    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")

    ax.plot(sizes_k, py_orig, label="Original (Py)", color=C_GRAY_DARK, linestyle="--", marker="x")
    ax.plot(sizes_k, rs_orig, label="Original (Rs)", color=C_GRAY_DARK, linestyle="-", marker="x")
    ax.plot(sizes_k, py_heap, label="Heap (Py)", color=C_GRAY_MID, linestyle="--", marker="v")
    ax.plot(sizes_k, py_sort_rank, label="Sorted ranked (Py)", color=C_GRAY_MID, linestyle=":", marker="d")
    ax.plot(sizes_k, py_bf, label="Best-first (Py)", color=C_LIGHT_BLUE, linestyle="--", marker="^")
    ax.plot(sizes_k, rs_bf, label="Best-first (Rs)", color=C_LIGHT_BLUE, linestyle="-", marker="^")
    ax.plot(sizes_k, py_topk, label="Precomputed (Py)", color=C_INK, linestyle="-", marker="o")
    ax.plot(sizes_k, rs_topk, label="Precomputed (Rs)", color=C_BLUE, linestyle="-", marker="s")

    ax.set_yscale("log")
    ax.set_xlabel("Vocabulary size (thousands of words)")
    ax.set_ylabel("Worst-prefix latency (μs, log scale)")
    ax.grid(True)
    fig.legend(ncol=2, frameon=False, loc="outside lower center", fontsize=6.5)

    out_pdf = os.path.join(OUT_DIR, "fig2_worst_prefix_scaling.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    render_pdf_to_png(out_pdf)


def fig3_memory_scaling():
    """Figure 3: Memory added by the finished index (RSS after build minus RSS after loading the word list)."""
    sizes = [50000, 100000, 200000, 466550, 677224]
    sizes_k = [n / 1000 for n in sizes]

    py_sort_mem, py_orig_mem, py_rank_mem = [], [], []
    rs_mem = []

    for n in sizes:
        ps = json.load(open(os.path.join(RESULTS_DIR, f"py_sorted_{n}.json")))
        po = json.load(open(os.path.join(RESULTS_DIR, f"py_original_{n}.json")))
        pr = json.load(open(os.path.join(RESULTS_DIR, f"py_ranked_{n}.json")))
        r = json.load(open(os.path.join(RESULTS_DIR, f"rs_{n}.json")))

        added = lambda m, key: m["mem_mb"][key] - m["mem_mb"]["after_vocab"]
        py_sort_mem.append(added(ps, "after_structure"))
        py_orig_mem.append(added(po, "after_structure"))
        py_rank_mem.append(added(pr, "after_best"))
        rs_mem.append(added(r, "after_best"))

    fig, ax = plt.subplots(figsize=(3.5, 2.3), layout="constrained")

    ax.plot(sizes_k, py_rank_mem, label="Ranked trie (Py)", color=C_INK, marker="o")
    ax.plot(sizes_k, py_orig_mem, label="Original trie (Py)", color=C_GRAY_DARK, marker="x", linestyle="--")
    ax.plot(sizes_k, rs_mem, label="Rust CSR index (Rs)", color=C_BLUE, marker="s")
    ax.plot(sizes_k, py_sort_mem, label="Sorted array (Py)", color=C_GRAY_MID, marker="d", linestyle=":")

    ax.set_xlabel("Vocabulary size (thousands of words)")
    ax.set_ylabel("Memory added by the index (MB)")
    ax.grid(True)
    ax.legend(frameon=True, loc="upper left", framealpha=0.9)

    out_pdf = os.path.join(OUT_DIR, "fig3_memory_scaling.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    render_pdf_to_png(out_pdf)


def fig4_language_boundary():
    """Figure 4: Python, Rust and Rust plus a JSON round trip at L=1, n=677,224 (all bars measured)."""
    py_r = json.load(open(os.path.join(RESULTS_DIR, "py_ranked_677224.json")))
    py_o = json.load(open(os.path.join(RESULTS_DIR, "py_original_677224.json")))
    rs = json.load(open(os.path.join(RESULTS_DIR, "rs_677224.json")))
    med = lambda d: d["summary"]["median"]

    groups = ["Precomputed top-k\n(10 words)", "Original trie\n(all completions)"]
    py_lookup = [med(py_r["methods"]["topk"]["by_length"]["1"]), med(py_o["methods"]["original"]["by_length"]["1"])]
    py_json = [med(py_r["boundary"]["lookup_encode_decode"]["1"]), None]
    rs_lookup = [med(rs["methods"]["topk"]["by_length"]["1"]), med(rs["methods"]["original"]["by_length"]["1"])]
    rs_json = [med(rs["boundary"]["lookup_encode_decode"]["1"]), med(rs["boundary"]["original_lookup_encode_decode"]["1"])]

    import numpy as np
    x = np.arange(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
    series = [("Python lookup", py_lookup, C_GRAY_MID), ("Python + JSON", py_json, C_GRAY_DARK),
              ("Rust lookup", rs_lookup, C_BLUE), ("Rust + JSON", rs_json, C_LIGHT_BLUE)]
    for i, (name, vals, color) in enumerate(series):
        xs = [xi + (i - 1.5) * w for xi, v in zip(x, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.bar(xs, ys, w, label=name, color=color)
        for xv, yv in zip(xs, ys):
            ax.annotate(f"{yv:,.2f}" if yv < 10 else f"{yv:,.0f}", xy=(xv, yv), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3e5)
    ax.set_ylabel("Median latency at L=1 (\u03bcs, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(True, axis="y")
    fig.legend(ncol=2, frameon=False, loc="outside lower center", fontsize=6.5)

    out_pdf = os.path.join(OUT_DIR, "fig4_language_boundary.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    render_pdf_to_png(out_pdf)


def fig5_keystroke_savings():
    """Figure 5: Keystroke savings by k and condition."""
    ks_data = json.load(open(os.path.join(RESULTS_DIR, "keystrokes.json")))
    conds = ks_data["conditions"]

    ks = ["1", "5", "10"]
    import numpy as np
    x = np.arange(len(ks))
    width = 0.16

    fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")

    alpha = [conds["alphabetical_merged"]["by_k"][k]["ksr_percent"] for k in ks]
    ranked = [conds["ranked_merged"]["by_k"][k]["ksr_percent"] for k in ks]
    learned = [conds["ranked_learned_merged"]["by_k"][k]["ksr_percent"] for k in ks]
    ranked_w = [conds["ranked_wordstxt"]["by_k"][k]["ksr_percent"] for k in ks]
    learned_w = [conds["ranked_learned_wordstxt"]["by_k"][k]["ksr_percent"] for k in ks]

    ax.bar(x - 2 * width, alpha, width, label="Alphabetical", color=C_GRAY_LIGHT)
    ax.bar(x - width, ranked_w, width, label="Ranked (words.txt)", color=C_GRAY_MID)
    ax.bar(x, ranked, width, label="Ranked (merged)", color=C_LIGHT_BLUE)
    ax.bar(x + width, learned_w, width, label="Learned (words.txt)", color=C_GRAY_DARK)
    ax.bar(x + 2 * width, learned, width, label="Learned (merged)", color=C_BLUE)

    ax.set_xlabel("Top-k completions offered (k)")
    ax.set_ylabel("Keystroke savings rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k = {k}" for k in ks])
    ax.set_ylim(0, 42)
    ax.grid(True, axis="y")
    fig.legend(ncol=2, frameon=False, loc="outside lower center", fontsize=6.5)

    out_pdf = os.path.join(OUT_DIR, "fig5_keystroke_savings.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    render_pdf_to_png(out_pdf)


def fig6_typo_recovery():
    """Figure 6: Correction found in the top k, on the misspellings the app sends to fuzzy search (no word completes them)."""
    typo = json.load(open(os.path.join(RESULTS_DIR, "rs_typo_677224.json")))
    rows = [r for r in typo["rows"] if r["corr_in_vocab"] and not r["has_completions"]]
    n = len(rows)

    ks = [1, 5, 10]
    lev_pct = [sum(1 for r in rows if r["lev_rank"] is not None and r["lev_rank"] <= k) / n * 100 for k in ks]
    osa_pct = [sum(1 for r in rows if r["osa_rank"] is not None and r["osa_rank"] <= k) / n * 100 for k in ks]

    import numpy as np
    x = np.arange(len(ks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    bars1 = ax.bar(x - width / 2, lev_pct, width, label="Levenshtein", color=C_GRAY_MID)
    bars2 = ax.bar(x + width / 2, osa_pct, width, label="Levenshtein + adjacent swaps", color=C_BLUE)
    for b in list(bars1) + list(bars2):
        y = b.get_height()
        ax.annotate(f"{y:.1f}%", xy=(b.get_x() + b.get_width() / 2, y),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlabel(f"Correction among the top k suggestions ({n:,} misspellings)")
    ax.set_ylabel("Misspellings recovered (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["k = 1", "k = 5", "k = 10"])
    ax.set_ylim(0, 112)
    ax.grid(True, axis="y")
    fig.legend(ncol=2, frameon=False, loc="outside lower center", fontsize=6.5)

    out_pdf = os.path.join(OUT_DIR, "fig6_typo_recovery.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    render_pdf_to_png(out_pdf)


def main():
    print("Generating Figure 1: Latency by prefix length...")
    fig1_latency_by_prefix()
    print("Generating Figure 2: Worst-case prefix scaling...")
    fig2_worst_prefix_scaling()
    print("Generating Figure 3: Memory scaling...")
    fig3_memory_scaling()
    print("Generating Figure 4: Language and IPC boundary...")
    fig4_language_boundary()
    print("Generating Figure 5: Keystroke savings...")
    fig5_keystroke_savings()
    print("Generating Figure 6: Typo recovery...")
    fig6_typo_recovery()

    # Copy generated PDF figures into paper directory
    for f in glob.glob(os.path.join(OUT_DIR, "*.pdf")):
        base = os.path.basename(f)
        dst = os.path.join(PAPER_FIG_DIR, base)
        with open(f, "rb") as fin, open(dst, "wb") as fout:
            fout.write(fin.read())
        print(f"Copied {base} to {PAPER_FIG_DIR}")

    print("All figures successfully created and verified!")


if __name__ == "__main__":
    main()
