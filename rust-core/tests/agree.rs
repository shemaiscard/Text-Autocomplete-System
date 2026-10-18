//! The Rust port must agree with brute force (and so with the Python core).
use autocomplete_core::*;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn random_vocab(r: &mut Lcg, n: usize, alphabet: &[u8], max_len: u64) -> Vec<(String, u64)> {
    let weights = [1u64, 2, 3, 5, 8, 50];
    let mut set = std::collections::BTreeMap::new();
    while set.len() < n {
        let len = 1 + r.below(max_len) as usize;
        let w: String = (0..len).map(|_| alphabet[r.below(alphabet.len() as u64) as usize] as char).collect();
        let wt = weights[r.below(6) as usize];
        set.entry(w).or_insert(wt);
    }
    set.into_iter().collect()
}

fn brute(v: &[(String, u64)], prefix: &str, k: usize) -> Vec<String> {
    let mut f: Vec<&(String, u64)> = v.iter().filter(|(w, _)| w.starts_with(prefix)).collect();
    f.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    f.into_iter().take(k).map(|(w, _)| w.clone()).collect()
}

fn osa(a: &[u8], b: &[u8]) -> u32 {
    let mut d = vec![vec![0u32; b.len() + 1]; a.len() + 1];
    for i in 0..=a.len() { d[i][0] = i as u32; }
    for j in 0..=b.len() { d[0][j] = j as u32; }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let mut v = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]) as u32);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = v;
        }
    }
    d[a.len()][b.len()]
}

#[test]
fn all_methods_match_brute_force() {
    let mut r = Lcg(7);
    for _ in 0..20 {
        let v = random_vocab(&mut r, 300, b"abcde", 7);
        for &k in &[1usize, 3, 10] {
            let ix = Index::with_caches(v.clone(), k);
            let mut prefixes: Vec<String> = vec!["".into(), "a".into(), "ab".into(), "eee".into(), "q".into()];
            prefixes.extend(v.iter().map(|(w, _)| w[..w.len().min(2)].to_string()));
            for p in &prefixes {
                let want = brute(&v, p, k);
                assert_eq!(ix.suggest(p, k), want, "topk {p} {k}");
                assert_eq!(ix.suggest_bestfirst(p, k), want, "bestfirst {p} {k}");
                assert_eq!(ix.suggest_heap(p, k), want, "heap {p} {k}");
                assert_eq!(ix.sorted_suggest_ranked(p, k), want, "sorted ranked {p} {k}");
                let mut all: Vec<String> = v.iter().map(|(w, _)| w.clone()).filter(|w| w.starts_with(p.as_str())).collect();
                all.sort();
                let got_all = { let mut g = ix.complete_all(p); g.sort(); g };
                assert_eq!(got_all, all);
                assert_eq!(ix.sorted_suggest(p, k), all.iter().take(k).cloned().collect::<Vec<_>>());
            }
        }
    }
}

#[test]
fn learning_matches_rebuild() {
    let mut r = Lcg(11);
    for _ in 0..15 {
        let v = random_vocab(&mut r, 250, b"abcde", 7);
        let k = [1usize, 3, 5, 10][r.below(4) as usize];
        let mut live = Index::with_caches(v.clone(), k);
        for _ in 0..40 {
            let w = v[r.below(v.len() as u64) as usize].0.clone();
            live.learn(&w, [1u64, 4, 100][r.below(3) as usize]).unwrap();
        }
        let learned: Vec<(String, u64)> = live.words.iter().cloned().zip(live.weight.iter().cloned()).collect();
        let fresh = Index::with_caches(learned.clone(), k);
        let mut prefixes: Vec<String> = vec!["".into()];
        for (w, _) in &v {
            for l in 1..=w.len() { prefixes.push(w[..l].to_string()); }
        }
        for p in &prefixes {
            assert_eq!(live.suggest(p, k), fresh.suggest(p, k), "{p}");
            assert_eq!(live.suggest_bestfirst(p, k), brute(&learned, p, k), "{p}");
        }
    }
    let mut ix = Index::with_caches(vec![("abc".into(), 1)], 2);
    assert!(ix.learn("ab", 5).is_err());
    assert!(ix.learn("abcd", 5).is_err());
    assert!(ix.learn("abc", 0).is_err());
}

#[test]
fn fuzzy_matches_brute_force() {
    let mut r = Lcg(3);
    let v = random_vocab(&mut r, 400, b"abcd", 6);
    let ix = Index::with_caches(v.clone(), 5);
    for _ in 0..60 {
        let len = 1 + r.below(7) as usize;
        let q: String = (0..len).map(|_| b"abcde"[r.below(5) as usize] as char).collect();
        for d in 1..=2u32 {
            let mut got: Vec<(u32, String)> = ix.fuzzy_search(&q, d, false).into_iter().map(|(x, id)| (x, ix.words[id as usize].clone())).collect();
            got.sort();
            let mut want: Vec<(u32, String)> = v.iter().map(|(w, _)| (levenshtein(q.as_bytes(), w.as_bytes()), w.clone())).filter(|x| x.0 <= d).collect();
            want.sort();
            assert_eq!(got, want, "lev {q} {d}");
            let mut got: Vec<(u32, String)> = ix.fuzzy_search(&q, d, true).into_iter().map(|(x, id)| (x, ix.words[id as usize].clone())).collect();
            got.sort();
            let mut want: Vec<(u32, String)> = v.iter().map(|(w, _)| (osa(q.as_bytes(), w.as_bytes()), w.clone())).filter(|x| x.0 <= d).collect();
            want.sort();
            assert_eq!(got, want, "osa {q} {d}");
        }
    }
    let ix = Index::with_caches(vec![("search".into(), 1000), ("perch".into(), 10), ("starch".into(), 500), ("serge".into(), 2)], 3);
    assert_eq!(ix.fuzzy_suggest("serch", 3), vec!["search", "perch", "starch"]);
    let (s, fz) = ix.suggest_or_fuzzy("serch", 3);
    assert!(fz && s[0].word == "search");
}

#[test]
fn boundary_round_trip() {
    let ix = Index::with_caches(vec![("the".into(), 9), ("then".into(), 3), ("they".into(), 5)], 10);
    let s = ix.suggest_scored("th", 10);
    let json = serde_json::to_string(&s).unwrap();
    let back: Vec<SuggestionOwned> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.iter().map(|x| x.word.as_str()).collect::<Vec<_>>(), vec!["the", "they", "then"]);
    assert_eq!(back[0].score, 9);
}

#[test]
fn merged_vocabulary_and_case_rule() {
    let dir = std::env::temp_dir().join(format!("acore-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wp = dir.join("w.txt");
    let fp = dir.join("f.tsv");
    std::fs::write(&wp, "OF\ncat\n").unwrap();
    std::fs::write(&fp, "# c\nof\t900\ncat\t7\nweb\t3\n").unwrap();
    let a = load_vocabulary(wp.to_str().unwrap(), fp.to_str().unwrap(), false).unwrap();
    assert_eq!(a, vec![("OF".to_string(), 900), ("cat".to_string(), 7)]);
    let b = load_vocabulary(wp.to_str().unwrap(), fp.to_str().unwrap(), true).unwrap();
    assert_eq!(b, vec![("OF".to_string(), 1), ("cat".to_string(), 7), ("of".to_string(), 900), ("web".to_string(), 3)]);
    std::fs::remove_dir_all(&dir).unwrap();
}
