"""Sorted array with binary search (the leanest baseline)."""
import heapq
from bisect import bisect_left

_MAX_CHAR = "\U0010ffff"


class SortedIndex:
    def __init__(self, words, weights=None):
        self.words = sorted(words)          # O(n log n) presorting
        self.weight = weights if weights is not None else {}

    def range(self, prefix):
        """Index range [lo, hi) of the words that start with the prefix. O(L log n)."""
        a = self.words
        lo = bisect_left(a, prefix)
        hi = bisect_left(a, prefix + _MAX_CHAR, lo)
        return lo, hi

    def suggest(self, prefix, k=10):
        """First k matches in alphabetical order. O(L log n + k)."""
        lo, hi = self.range(prefix)
        return self.words[lo:min(hi, lo + k)]

    def suggest_ranked(self, prefix, k=10):
        """Top-k by weight; scans the whole range. O(L log n + m log k)."""
        lo, hi = self.range(prefix)
        W = self.weight
        return heapq.nsmallest(k, self.words[lo:hi], key=lambda w: (-W[w], w))
