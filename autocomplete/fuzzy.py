"""Typo-tolerant lookup: Levenshtein dynamic programming pruned by the trie.

One DP row is computed per trie node, so words that share a prefix share
the rows for that prefix. A branch is abandoned as soon as every entry of
its row exceeds the distance bound, because entries never decrease further
down. With transpositions=True the optimal string alignment distance is
used instead, where swapping two adjacent letters costs 1.
"""


def search(trie, text, max_dist, transpositions=False):
    """All (distance, word) pairs with distance(text, word) <= max_dist."""
    root = trie.root
    n = len(text)
    out = []
    if root.word is not None and n <= max_dist:
        out.append((n, root.word))
    if not root.children:
        return out
    first = list(range(n + 1))
    stack = [(child, ch, first, None, "") for ch, child in root.children.items()]
    pop, push = stack.pop, stack.append
    while stack:
        node, ch, prev, prev2, pch = pop()
        cur = [prev[0] + 1]
        row_min = cur[0]
        for j in range(1, n + 1):
            tj = text[j - 1]
            v = prev[j - 1] + (tj != ch)
            a = prev[j] + 1
            if a < v:
                v = a
            a = cur[j - 1] + 1
            if a < v:
                v = a
            if (transpositions and prev2 is not None and j > 1
                    and ch == text[j - 2] and pch == tj):
                a = prev2[j - 2] + 1
                if a < v:
                    v = a
            cur.append(v)
            if v < row_min:
                row_min = v
        if node.word is not None and cur[n] <= max_dist:
            out.append((cur[n], node.word))
        if row_min <= max_dist and node.children:
            for c2, child in node.children.items():
                push((child, c2, cur, prev, ch))
    return out


def suggest(trie, text, k=10, transpositions=False):
    """Closest words: distance 1 first, distance 2 only if fewer than k found.

    Ranked by (distance, weight descending, word).
    """
    W = trie.weight
    found = search(trie, text, 1, transpositions)
    if len(found) < k:
        found = search(trie, text, 2, transpositions)
    found.sort(key=lambda r: (r[0], -W[r[1]], r[1]))
    return [w for _, w in found[:k]]
