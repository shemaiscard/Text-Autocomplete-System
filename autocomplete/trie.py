"""Ranked prefix search over a character trie.

Every word has an integer weight: its corpus frequency per billion words
plus whatever the user taught it (see learning.py). All query methods
order results by (weight descending, word ascending), so for the same
prefix and k they return exactly the same list. They differ in cost:

    suggest            precomputed top-k list per node     O(L + k)
    suggest_bestfirst  best-first search on per-node best  O(L + k * s * log(k * s))
    suggest_heap       scan the subtree with a k-heap      O(L + m log k)

L is the prefix length, m the number of words under the prefix and s
the largest number of children a node has.
"""
import heapq

from .data import FLOOR_WEIGHT

DEFAULT_K = 10


class Node:
    __slots__ = ("children", "word", "top", "best")

    def __init__(self):
        self.children = None  # dict: character -> Node, created on demand
        self.word = None      # the word that ends here, or None
        self.top = ()         # top-k words of this subtree (precomputed)
        self.best = None      # best word of this subtree (for best-first)


class RankedTrie:
    """A trie whose nodes can cache their top-k words and their best word."""

    def __init__(self, words, weights, k=DEFAULT_K, cache_topk=True, cache_best=True):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.weight = weights
        self.root = Node()
        self.size = 0
        self.node_count = 1
        self.has_topk = False
        self.has_best = False
        for w in words:
            self.insert(w)
        if cache_topk:
            self.build_topk()
        if cache_best:
            self.build_best()

    # ----- construction -------------------------------------------------
    def insert(self, word, weight=None):
        """Add a word. Caches built earlier are dropped and must be rebuilt."""
        if weight is not None or word not in self.weight:
            self.weight[word] = weight if weight is not None else FLOOR_WEIGHT
        node = self.root
        for ch in word:
            c = node.children
            if c is None:
                c = node.children = {}
            nxt = c.get(ch)
            if nxt is None:
                nxt = c[ch] = Node()
                self.node_count += 1
            node = nxt
        if node.word is None:
            node.word = word
            self.size += 1
        if self.has_topk or self.has_best:
            self.has_topk = self.has_best = False

    def build_topk(self):
        """Store the k best words of every subtree in its root node. O(n_nodes * s * k log k)."""
        W = self.weight
        k = self.k

        def key(w):
            return (-W[w], w)

        def visit(node):
            ch = node.children
            if not ch:
                node.top = (node.word,)
                return node.top
            tops = [visit(c) for c in ch.values()]
            if node.word is None and len(tops) == 1:
                node.top = tops[0]          # a chain node shares its child's list
                return node.top
            cand = [node.word] if node.word is not None else []
            for t in tops:
                cand.extend(t)
            cand.sort(key=key)
            top = tuple(cand[:k])
            for t in tops:                  # share an equal child list
                if t == top:
                    top = t
                    break
            node.top = top
            return top

        if self.root.children:
            visit(self.root)
        self.has_topk = True

    def build_best(self):
        """Store the best word of every subtree in its root node. O(n_nodes)."""
        W = self.weight

        def visit(node):
            b = node.word
            if node.children:
                for c in node.children.values():
                    cb = visit(c)
                    if b is None or W[cb] > W[b] or (W[cb] == W[b] and cb < b):
                        b = cb
            node.best = b
            return b

        if self.root.children:
            visit(self.root)
        self.has_best = True

    # ----- queries ------------------------------------------------------
    def find(self, prefix):
        """The node reached by the prefix, or None. O(L)."""
        node = self.root
        for ch in prefix:
            c = node.children
            if c is None:
                return None
            node = c.get(ch)
            if node is None:
                return None
        return node

    def __contains__(self, word):
        node = self.find(word)
        return node is not None and node.word == word

    def suggest(self, prefix, k=None):
        """Top-k words under the prefix from the precomputed lists."""
        if k is None:
            k = self.k
        if k > self.k or not self.has_topk:
            return self.suggest_bestfirst(prefix, k)
        node = self.find(prefix)
        if node is None:
            return []
        top = node.top
        return list(top) if k >= len(top) else list(top[:k])

    def suggest_bestfirst(self, prefix, k=None):
        """Best-first search guided by each node's best word (Hsu and Ottaviano, 2013)."""
        if k is None:
            k = self.k
        if not self.has_best:
            self.build_best()
        node = self.find(prefix)
        if node is None or node.best is None:
            return []
        W = self.weight
        push, pop = heapq.heappush, heapq.heappop
        b = node.best
        # Entries are (-weight, word, is_node, node). The pair (-weight, word)
        # is unique among live entries, so nodes are never compared.
        heap = [(-W[b], b, 1, node)]
        out = []
        while heap and len(out) < k:
            _, w, is_node, nd = pop(heap)
            if not is_node:
                out.append(w)
                continue
            if nd.word is not None:
                push(heap, (-W[nd.word], nd.word, 0, None))
            if nd.children:
                for c in nd.children.values():
                    cb = c.best
                    push(heap, (-W[cb], cb, 1, c))
        return out

    def suggest_heap(self, prefix, k=None):
        """Visit every word under the prefix and keep the k best in a heap."""
        if k is None:
            k = self.k
        node = self.find(prefix)
        if node is None:
            return []
        W = self.weight
        return heapq.nsmallest(k, _iter_words(node), key=lambda w: (-W[w], w))

    def complete_all(self, prefix):
        """Every word under the prefix, unranked (what the first version showed)."""
        node = self.find(prefix)
        return [] if node is None else list(_iter_words(node))

    # ----- adaptive learning ---------------------------------------------
    def learn(self, word, delta):
        """Raise a word's weight by delta > 0 and repair the caches on its path.

        Only nodes on the root-to-word path can change. Walking up from the
        word, the update stops at the first node whose list the word does not
        enter, because no ancestor list can admit it either. Cost O(L * k).
        """
        if delta <= 0:
            raise ValueError("delta must be positive")
        path = [self.root]
        node = self.root
        for ch in word:
            c = node.children
            node = c.get(ch) if c else None
            if node is None:
                raise KeyError(word)
            path.append(node)
        if node.word != word:
            raise KeyError(word)
        W = self.weight
        W[word] += delta
        k = self.k

        def key(w):
            return (-W[w], w)

        if self.has_topk:
            kw = key(word)
            below = None
            for nd in reversed(path):
                top = nd.top
                if word in top:
                    new = tuple(sorted(top, key=key))
                elif len(top) < k or kw < key(top[-1]):
                    new = tuple(sorted(top[:k - 1] + (word,), key=key))
                else:
                    break
                if below is not None and below == new:
                    new = below
                nd.top = new
                below = new
        if self.has_best:
            for nd in reversed(path):
                b = nd.best
                if b == word or W[word] > W[b] or (W[word] == W[b] and word < b):
                    nd.best = word
                else:
                    break


def _iter_words(node):
    stack = [node]
    pop, extend = stack.pop, stack.extend
    while stack:
        nd = pop()
        if nd.word is not None:
            yield nd.word
        if nd.children:
            extend(nd.children.values())
