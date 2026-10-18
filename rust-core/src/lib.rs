//! Ranked prefix search over a byte trie, the Rust port of the Python core.
//!
//! A front end (for example a Tauri command) calls [`Index::suggest_scored`]
//! once per keystroke pause and receives only the top-k words, so the whole
//! lookup stays inside Rust and only k short strings cross the boundary.
//!
//! Words get ids in byte order, and every result list is ordered by
//! (weight descending, id ascending), which equals the Python order
//! (weight descending, word ascending) for ASCII words.
//!
//! After construction the trie is frozen into compressed sparse rows: node
//! `v` owns edges `edge_start[v]..edge_start[v + 1]`, sorted by label.
//! Learning changes weights, never the shape, so the layout stays valid.

use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io;

pub const NONE: u32 = u32::MAX;
pub const DEFAULT_K: usize = 10;
/// Weight of a word that has no corpus frequency.
pub const FLOOR_WEIGHT: u64 = 1;
/// Weight added per pick (100,000 per billion words, Zipf 5).
pub const LEARN_DELTA: u64 = 100_000;

/// What a suggest call returns across the UI boundary.
#[derive(Debug, Clone, Serialize)]
pub struct Suggestion<'a> {
    pub word: &'a str,
    pub score: u64,
}

/// The same record after it crossed the boundary (owned strings).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct SuggestionOwned {
    pub word: String,
    pub score: u64,
}

pub struct Index {
    pub words: Vec<String>,
    pub weight: Vec<u64>,
    edge_start: Vec<u32>,
    edge_label: Vec<u8>,
    edge_child: Vec<u32>,
    node_word: Vec<u32>,
    k: usize,
    top_start: Vec<u32>,
    top_ids: Vec<u32>,
    best: Vec<u32>,
    has_topk: bool,
    has_best: bool,
}

#[derive(PartialEq, Eq)]
struct Entry {
    w: u64,
    id: u32,
    node: u32,
}

impl Ord for Entry {
    // Max-heap order: higher weight first, then smaller id.
    fn cmp(&self, o: &Self) -> Ordering {
        self.w
            .cmp(&o.w)
            .then_with(|| o.id.cmp(&self.id))
            .then_with(|| self.node.cmp(&o.node))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl Index {
    /// Build the trie (no caches yet). Duplicate words keep the first weight.
    pub fn new(mut pairs: Vec<(String, u64)>, k: usize) -> Index {
        assert!(k >= 1, "k must be at least 1");
        pairs.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        pairs.dedup_by(|a, b| a.0 == b.0);
        let (words, weight): (Vec<String>, Vec<u64>) = pairs.into_iter().unzip();
        // Sorted input: the child to follow is always the last one created.
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut node_word: Vec<u32> = vec![NONE];
        for (id, w) in words.iter().enumerate() {
            let mut v = 0usize;
            for &b in w.as_bytes() {
                match children[v].last() {
                    Some(&(l, c)) if l == b => v = c as usize,
                    _ => {
                        let c = children.len() as u32;
                        children[v].push((b, c));
                        children.push(Vec::new());
                        node_word.push(NONE);
                        v = c as usize;
                    }
                }
            }
            node_word[v] = id as u32;
        }
        let n = children.len();
        let total: usize = children.iter().map(|c| c.len()).sum();
        let mut edge_start = Vec::with_capacity(n + 1);
        let mut edge_label = Vec::with_capacity(total);
        let mut edge_child = Vec::with_capacity(total);
        for c in &children {
            edge_start.push(edge_label.len() as u32);
            for &(l, ch) in c {
                edge_label.push(l);
                edge_child.push(ch);
            }
        }
        edge_start.push(edge_label.len() as u32);
        drop(children);
        Index {
            words,
            weight,
            edge_start,
            edge_label,
            edge_child,
            node_word,
            k,
            top_start: Vec::new(),
            top_ids: Vec::new(),
            best: Vec::new(),
            has_topk: false,
            has_best: false,
        }
    }

    pub fn with_caches(pairs: Vec<(String, u64)>, k: usize) -> Index {
        let mut ix = Index::new(pairs, k);
        ix.build_topk();
        ix.build_best();
        ix
    }

    pub fn node_count(&self) -> usize {
        self.node_word.len()
    }
    pub fn len(&self) -> usize {
        self.words.len()
    }
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    fn edges(&self, v: usize) -> std::ops::Range<usize> {
        self.edge_start[v] as usize..self.edge_start[v + 1] as usize
    }

    #[inline]
    fn better(weight: &[u64], a: u32, b: u32) -> bool {
        let (wa, wb) = (weight[a as usize], weight[b as usize]);
        wa > wb || (wa == wb && a < b)
    }

    /// Store the k best words of every subtree. Nodes are numbered in
    /// preorder, so a reverse scan visits children before parents.
    pub fn build_topk(&mut self) {
        let n = self.node_count();
        let mut cnt = vec![0u32; n];
        for v in (0..n).rev() {
            let mut c = (self.node_word[v] != NONE) as u32;
            for e in self.edges(v) {
                c += cnt[self.edge_child[e] as usize];
            }
            cnt[v] = c;
        }
        let mut top_start = Vec::with_capacity(n + 1);
        let mut acc = 0u32;
        for &c in &cnt {
            top_start.push(acc);
            acc += c.min(self.k as u32);
        }
        top_start.push(acc);
        drop(cnt);
        let mut top_ids = vec![NONE; acc as usize];
        let weight = &self.weight;
        let mut cand: Vec<u32> = Vec::new();
        for v in (0..n).rev() {
            cand.clear();
            if self.node_word[v] != NONE {
                cand.push(self.node_word[v]);
            }
            for e in self.edges(v) {
                let c = self.edge_child[e] as usize;
                cand.extend_from_slice(&top_ids[top_start[c] as usize..top_start[c + 1] as usize]);
            }
            cand.sort_unstable_by(|&a, &b| weight[b as usize].cmp(&weight[a as usize]).then(a.cmp(&b)));
            let (s, e) = (top_start[v] as usize, top_start[v + 1] as usize);
            top_ids[s..e].copy_from_slice(&cand[..e - s]);
        }
        self.top_start = top_start;
        self.top_ids = top_ids;
        self.has_topk = true;
    }

    /// Store the best word of every subtree (for best-first search).
    pub fn build_best(&mut self) {
        let n = self.node_count();
        let mut best = vec![NONE; n];
        for v in (0..n).rev() {
            let mut b = self.node_word[v];
            for e in self.edges(v) {
                let cb = best[self.edge_child[e] as usize];
                if b == NONE || Self::better(&self.weight, cb, b) {
                    b = cb;
                }
            }
            best[v] = b;
        }
        self.best = best;
        self.has_best = true;
    }

    /// Node reached by the prefix. O(L log s).
    pub fn find(&self, prefix: &str) -> Option<usize> {
        let mut v = 0usize;
        for &b in prefix.as_bytes() {
            let r = self.edges(v);
            let s = r.start;
            match self.edge_label[r].binary_search(&b) {
                Ok(i) => v = self.edge_child[s + i] as usize,
                Err(_) => return None,
            }
        }
        Some(v)
    }

    pub fn contains(&self, word: &str) -> Option<u32> {
        self.find(word).map(|v| self.node_word[v]).filter(|&id| id != NONE)
    }

    fn top_slice(&self, v: usize) -> &[u32] {
        &self.top_ids[self.top_start[v] as usize..self.top_start[v + 1] as usize]
    }

    /// Top-k ids from the precomputed lists. O(L log s + k).
    pub fn suggest_ids(&self, prefix: &str, k: usize) -> Vec<u32> {
        if !self.has_topk || k > self.k {
            return self.suggest_bestfirst_ids(prefix, k);
        }
        match self.find(prefix) {
            None => Vec::new(),
            Some(v) => {
                let t = self.top_slice(v);
                t[..k.min(t.len())].to_vec()
            }
        }
    }

    pub fn suggest(&self, prefix: &str, k: usize) -> Vec<&str> {
        if !self.has_topk || k > self.k {
            return self.ids_to_words(&self.suggest_bestfirst_ids(prefix, k));
        }
        match self.find(prefix) {
            None => Vec::new(),
            Some(v) => {
                let t = self.top_slice(v);
                t[..k.min(t.len())].iter().map(|&id| self.words[id as usize].as_str()).collect()
            }
        }
    }

    /// The call a UI makes per keystroke pause: top-k with scores.
    pub fn suggest_scored(&self, prefix: &str, k: usize) -> Vec<Suggestion<'_>> {
        self.suggest_ids(prefix, k)
            .into_iter()
            .map(|id| Suggestion { word: &self.words[id as usize], score: self.weight[id as usize] })
            .collect()
    }

    /// Top-k, or close spellings when nothing completes the prefix.
    pub fn suggest_or_fuzzy(&self, prefix: &str, k: usize) -> (Vec<Suggestion<'_>>, bool) {
        let found = self.suggest_scored(prefix, k);
        if !found.is_empty() || prefix.is_empty() {
            return (found, false);
        }
        let ids = self.fuzzy_suggest_ids(prefix, k, false);
        let out = ids
            .into_iter()
            .map(|id| Suggestion { word: &self.words[id as usize], score: self.weight[id as usize] })
            .collect();
        (out, true)
    }

    pub fn ids_to_words(&self, ids: &[u32]) -> Vec<&str> {
        ids.iter().map(|&id| self.words[id as usize].as_str()).collect()
    }

    /// Best-first search guided by each node's best word (Hsu and Ottaviano, 2013).
    pub fn suggest_bestfirst_ids(&self, prefix: &str, k: usize) -> Vec<u32> {
        assert!(self.has_best, "build_best() first");
        if k == 0 {
            return Vec::new();
        }
        let Some(v) = self.find(prefix) else { return Vec::new() };
        let b = self.best[v];
        if b == NONE {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(4 * k + 32);
        heap.push(Entry { w: self.weight[b as usize], id: b, node: v as u32 });
        let mut out = Vec::with_capacity(k);
        while let Some(Entry { id, node, .. }) = heap.pop() {
            if node == NONE {
                out.push(id);
                if out.len() == k {
                    break;
                }
                continue;
            }
            let nv = node as usize;
            let wd = self.node_word[nv];
            if wd != NONE {
                heap.push(Entry { w: self.weight[wd as usize], id: wd, node: NONE });
            }
            for e in self.edges(nv) {
                let c = self.edge_child[e];
                let cb = self.best[c as usize];
                heap.push(Entry { w: self.weight[cb as usize], id: cb, node: c });
            }
        }
        out
    }

    pub fn suggest_bestfirst(&self, prefix: &str, k: usize) -> Vec<&str> {
        self.ids_to_words(&self.suggest_bestfirst_ids(prefix, k))
    }

    /// Visit every word under the prefix, keep the k best in a heap. O(L + m log k).
    pub fn suggest_heap(&self, prefix: &str, k: usize) -> Vec<&str> {
        let Some(v) = self.find(prefix) else { return Vec::new() };
        if k == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<(Reverse<u64>, u32)> = BinaryHeap::with_capacity(k + 1);
        let mut stack = vec![v as u32];
        while let Some(x) = stack.pop() {
            let x = x as usize;
            let id = self.node_word[x];
            if id != NONE {
                let key = (Reverse(self.weight[id as usize]), id);
                if heap.len() < k {
                    heap.push(key);
                } else if let Some(mut worst) = heap.peek_mut() {
                    if key < *worst {
                        *worst = key;
                    }
                }
            }
            for e in self.edges(x) {
                stack.push(self.edge_child[e]);
            }
        }
        heap.into_sorted_vec().into_iter().map(|(_, id)| self.words[id as usize].as_str()).collect()
    }

    /// The first version's algorithm: every word under the prefix, unranked.
    pub fn complete_all(&self, prefix: &str) -> Vec<String> {
        let Some(v) = self.find(prefix) else { return Vec::new() };
        let mut out = Vec::new();
        let mut buf = prefix.as_bytes().to_vec();
        self.collect(v, &mut buf, &mut out);
        out
    }

    fn collect(&self, v: usize, buf: &mut Vec<u8>, out: &mut Vec<String>) {
        if self.node_word[v] != NONE {
            out.push(String::from_utf8(buf.clone()).expect("utf-8 word"));
        }
        for e in self.edges(v) {
            buf.push(self.edge_label[e]);
            self.collect(self.edge_child[e] as usize, buf, out);
            buf.pop();
        }
    }

    /// Sorted-array range of the prefix (words are kept in byte order). O(L log n).
    pub fn sorted_range(&self, prefix: &str) -> (usize, usize) {
        let p = prefix.as_bytes();
        let lo = self.words.partition_point(|w| w.as_bytes() < p);
        let hi = lo + self.words[lo..].partition_point(|w| w.as_bytes().starts_with(p));
        (lo, hi)
    }

    /// Sorted array + binary search, first k in alphabetical order.
    pub fn sorted_suggest(&self, prefix: &str, k: usize) -> Vec<&str> {
        let (lo, hi) = self.sorted_range(prefix);
        self.words[lo..hi.min(lo + k)].iter().map(|s| s.as_str()).collect()
    }

    /// Sorted array, top-k by weight: scans the whole range with a k-heap.
    pub fn sorted_suggest_ranked(&self, prefix: &str, k: usize) -> Vec<&str> {
        let (lo, hi) = self.sorted_range(prefix);
        if k == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<(Reverse<u64>, u32)> = BinaryHeap::with_capacity(k + 1);
        for id in lo as u32..hi as u32 {
            let key = (Reverse(self.weight[id as usize]), id);
            if heap.len() < k {
                heap.push(key);
            } else if let Some(mut worst) = heap.peek_mut() {
                if key < *worst {
                    *worst = key;
                }
            }
        }
        heap.into_sorted_vec().into_iter().map(|(_, id)| self.words[id as usize].as_str()).collect()
    }

    /// Raise a word's weight and repair the caches on its path only. O(L * k).
    pub fn learn(&mut self, word: &str, delta: u64) -> Result<(), String> {
        if delta == 0 {
            return Err("delta must be positive".into());
        }
        let mut path = Vec::with_capacity(word.len() + 1);
        let mut v = 0usize;
        path.push(v);
        for &b in word.as_bytes() {
            let r = self.edges(v);
            let s = r.start;
            match self.edge_label[r].binary_search(&b) {
                Ok(i) => v = self.edge_child[s + i] as usize,
                Err(_) => return Err(format!("unknown word {word}")),
            }
            path.push(v);
        }
        let id = self.node_word[v];
        if id == NONE {
            return Err(format!("unknown word {word}"));
        }
        self.weight[id as usize] += delta;
        let weight = &self.weight;
        if self.has_topk {
            for &x in path.iter().rev() {
                let (s, e) = (self.top_start[x] as usize, self.top_start[x + 1] as usize);
                let list = &mut self.top_ids[s..e];
                let mut i = match list.iter().position(|&y| y == id) {
                    Some(p) => p,
                    None => {
                        let last = list.len() - 1;
                        if Self::better(weight, id, list[last]) {
                            list[last] = id;
                            last
                        } else {
                            break; // no ancestor list can admit the word either
                        }
                    }
                };
                while i > 0 && Self::better(weight, id, list[i - 1]) {
                    list.swap(i, i - 1);
                    i -= 1;
                }
            }
        }
        if self.has_best {
            for &x in path.iter().rev() {
                let b = self.best[x];
                if b == id || Self::better(weight, id, b) {
                    self.best[x] = id;
                } else {
                    break;
                }
            }
        }
        Ok(())
    }

    /// All (distance, id) with Levenshtein (or OSA) distance <= max_d,
    /// computing one DP row per trie node and pruning hopeless branches.
    pub fn fuzzy_search(&self, text: &str, max_d: u32, transpositions: bool) -> Vec<(u32, u32)> {
        let t = text.as_bytes();
        let n = t.len();
        let w = n + 1;
        let mut out = Vec::new();
        if self.node_word[0] != NONE && n as u32 <= max_d {
            out.push((n as u32, self.node_word[0]));
        }
        let mut rows: Vec<u32> = (0..w as u32).collect();
        let mut labels: Vec<u8> = vec![0];
        let mut stack: Vec<(u32, u32, u8)> = Vec::new();
        for e in self.edges(0) {
            stack.push((self.edge_child[e], 1, self.edge_label[e]));
        }
        while let Some((v, d, ch)) = stack.pop() {
            let d = d as usize;
            if rows.len() < (d + 1) * w {
                rows.resize((d + 1) * w, 0);
                labels.resize(d + 1, 0);
            }
            labels[d] = ch;
            let (before, cur) = rows.split_at_mut(d * w);
            let prev = &before[(d - 1) * w..];
            let cur = &mut cur[..w];
            cur[0] = prev[0] + 1;
            let mut row_min = cur[0];
            for j in 1..w {
                let tj = t[j - 1];
                let mut val = prev[j - 1] + (tj != ch) as u32;
                val = val.min(prev[j] + 1).min(cur[j - 1] + 1);
                if transpositions && d >= 2 && j >= 2 && ch == t[j - 2] && labels[d - 1] == tj {
                    val = val.min(before[(d - 2) * w + j - 2] + 1);
                }
                cur[j] = val;
                row_min = row_min.min(val);
            }
            let id = self.node_word[v as usize];
            if id != NONE && cur[n] <= max_d {
                out.push((cur[n], id));
            }
            if row_min <= max_d {
                for e in self.edges(v as usize) {
                    stack.push((self.edge_child[e], d as u32 + 1, self.edge_label[e]));
                }
            }
        }
        out
    }

    /// Distance 1 first, distance 2 only if fewer than k were found;
    /// ranked by (distance, weight descending, id).
    pub fn fuzzy_suggest_ids(&self, text: &str, k: usize, transpositions: bool) -> Vec<u32> {
        let mut found = self.fuzzy_search(text, 1, transpositions);
        if found.len() < k {
            found = self.fuzzy_search(text, 2, transpositions);
        }
        let weight = &self.weight;
        found.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0).then(weight[b.1 as usize].cmp(&weight[a.1 as usize])).then(a.1.cmp(&b.1))
        });
        found.truncate(k);
        found.into_iter().map(|(_, id)| id).collect()
    }

    pub fn fuzzy_suggest(&self, text: &str, k: usize) -> Vec<&str> {
        self.ids_to_words(&self.fuzzy_suggest_ids(text, k, false))
    }
}

/// Plain Levenshtein distance (two-row DP), for brute-force comparison.
pub fn levenshtein(a: &[u8], b: &[u8]) -> u32 {
    let mut prev: Vec<u32> = (0..=b.len() as u32).collect();
    let mut cur = vec![0u32; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i as u32;
        for j in 1..=b.len() {
            let sub = prev[j - 1] + (a[i - 1] != b[j - 1]) as u32;
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Lines of "word<TAB>weight" ('#' lines are comments).
pub fn load_vocab_tsv(path: &str) -> io::Result<Vec<(String, u64)>> {
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((w, c)) = line.split_once('\t') {
            if let Ok(c) = c.trim().parse::<u64>() {
                out.push((w.to_string(), c));
            }
        }
    }
    Ok(out)
}

pub fn load_words(path: &str) -> io::Result<Vec<String>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect())
}

pub fn load_frequencies(path: &str) -> io::Result<HashMap<String, u64>> {
    Ok(load_vocab_tsv(path)?.into_iter().collect())
}

/// Corpus weight for every word. A frequency belongs to the lowercase form;
/// a word with capitals takes it only when its lowercase form is not itself
/// in the vocabulary (same rule as the Python core).
pub fn base_weights(words: Vec<String>, freqs: &HashMap<String, u64>) -> Vec<(String, u64)> {
    let present: HashSet<&str> = words.iter().map(|w| w.as_str()).collect();
    let weights: Vec<u64> = words
        .iter()
        .map(|w| {
            let low = w.to_lowercase();
            if low == *w || !present.contains(low.as_str()) {
                freqs.get(&low).copied().unwrap_or(FLOOR_WEIGHT)
            } else {
                FLOOR_WEIGHT
            }
        })
        .collect();
    drop(present);
    words.into_iter().zip(weights).collect()
}

/// words.txt plus (when merge is true) every frequency-file word it lacks.
pub fn load_vocabulary(words_path: &str, freq_path: &str, merge: bool) -> io::Result<Vec<(String, u64)>> {
    let mut words = load_words(words_path)?;
    let freqs = load_frequencies(freq_path)?;
    if merge {
        let have: HashSet<String> = words.iter().cloned().collect();
        let mut extra: Vec<String> = freqs.keys().filter(|w| !have.contains(*w)).cloned().collect();
        extra.sort();
        words.extend(extra);
    }
    Ok(base_weights(words, &freqs))
}
