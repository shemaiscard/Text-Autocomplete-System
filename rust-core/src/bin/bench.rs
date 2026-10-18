//! Rust benchmark, same inputs and JSON layout as bench/bench_python.py.
//!
//! cargo run --release --bin bench -- --cache DIR --n 466550 --out r.json [--full]
//! cargo run --release --bin bench -- --cache DIR --n 466550 --typo-eval out.json

use autocomplete_core::*;
use serde_json::{json, Map, Value};
use std::hint::black_box;
use std::time::Instant;

fn rss_mb(key: &str) -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let kb: f64 = rest.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(f64::NAN);
            return kb / 1024.0;
        }
    }
    f64::NAN
}

fn summarize(xs: &[f64]) -> Value {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let median = if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 };
    let mean = s.iter().sum::<f64>() / n as f64;
    let sd = if n > 1 { (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt() } else { 0.0 };
    json!({"n": n, "min": s[0], "median": median, "mean": mean, "sd": sd,
           "p95": s[(n - 1).min((0.95 * n as f64) as usize)], "max": s[n - 1]})
}

/// Per-call microseconds: `samples` warm samples, each the mean of `reps`
/// back-to-back calls, reps chosen so a sample lasts at least target_ns.
fn sample_calls<T, F: FnMut() -> T>(mut f: F, samples: usize, target_ns: u128) -> (Vec<f64>, u64) {
    black_box(f());
    let mut reps: u64 = 1;
    loop {
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(f());
        }
        let el = t0.elapsed().as_nanos();
        if el >= target_ns || reps >= 2_000_000 {
            break;
        }
        let want = (reps as f64 * target_ns as f64 / (el.max(1) as f64)) as u64 + 1;
        reps = (reps * 2).max(want).min(2_000_000);
    }
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(f());
        }
        out.push(t0.elapsed().as_nanos() as f64 / reps as f64 / 1000.0);
    }
    (out, reps)
}

fn median_of(xs: &[f64]) -> f64 {
    summarize(xs)["median"].as_f64().unwrap()
}

struct Args {
    cache: String,
    n: usize,
    k: usize,
    samples: usize,
    out: String,
    full: bool,
    typo: Option<String>,
}

fn args() -> Args {
    let v: Vec<String> = std::env::args().collect();
    let get = |name: &str| v.iter().position(|x| x == name).map(|i| v[i + 1].clone());
    Args {
        cache: get("--cache").expect("--cache"),
        n: get("--n").expect("--n").parse().unwrap(),
        k: get("--k").map(|x| x.parse().unwrap()).unwrap_or(10),
        samples: get("--samples").map(|x| x.parse().unwrap()).unwrap_or(15),
        out: get("--out").unwrap_or_default(),
        full: v.iter().any(|x| x == "--full"),
        typo: get("--typo-eval"),
    }
}

type Prefixes = Vec<(String, Vec<(String, u64)>)>;

fn read_prefixes(path: &str) -> (Prefixes, Vec<(String, String, u64)>) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut typed = Vec::new();
    for l in 1..=6 {
        let key = l.to_string();
        let items = v["typed"][&key]
            .as_array()
            .unwrap()
            .iter()
            .map(|it| (it["prefix"].as_str().unwrap().to_string(), it["count"].as_u64().unwrap()))
            .collect();
        typed.push((key, items));
    }
    let mut worst = Vec::new();
    for l in 1..=6 {
        let key = l.to_string();
        let it = &v["worst"][&key];
        worst.push((key, it["prefix"].as_str().unwrap().to_string(), it["count"].as_u64().unwrap()));
    }
    (typed, worst)
}

fn time_method<T, F: Fn(&str, usize) -> T>(f: &F, k: usize, typed: &Prefixes, samples: usize) -> Value {
    let mut out = Map::new();
    for (l, items) in typed {
        let mut pooled = Vec::new();
        let mut per = Vec::new();
        for (p, count) in items {
            let (xs, reps) = sample_calls(|| f(black_box(p.as_str()), k), samples, 200_000);
            pooled.extend_from_slice(&xs);
            per.push(json!({"prefix": p, "count": count, "reps": reps, "median_us": median_of(&xs), "samples_us": xs}));
        }
        out.insert(l.clone(), json!({"summary": summarize(&pooled), "per_prefix": per}));
    }
    Value::Object(out)
}

fn worst_cases<T, F: Fn(&str, usize) -> T>(f: &F, k: usize, worst: &[(String, String, u64)], samples: usize) -> Value {
    let mut out = Map::new();
    for (l, p, count) in worst {
        let (xs, reps) = sample_calls(|| f(black_box(p.as_str()), k), samples, 200_000);
        out.insert(l.clone(), json!({"prefix": p, "count": count, "reps": reps, "summary": summarize(&xs), "samples_us": xs}));
    }
    Value::Object(out)
}

fn env() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .map(|l| l.split(':').nth(1).unwrap().trim().to_string());
    let rustc = std::process::Command::new("rustc").arg("--version").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    json!({"cpu": cpu, "logical_cpus": std::thread::available_parallelism().map(|x| x.get()).ok(),
           "rustc": rustc, "profile": "release (opt-level 3, codegen-units 1)",
           "governor": std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").ok().map(|s| s.trim().to_string()),
           "loadavg_start": load.trim()})
}

fn read_lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path).unwrap().lines().filter(|l| !l.is_empty()).map(String::from).collect()
}

fn typo_eval(ix: &Index, cache: &str, out: &str) {
    let mut rows = Vec::new();
    for line in read_lines(&format!("{cache}/misspellings.tsv")) {
        let (m, corr) = line.split_once('\t').unwrap();
        let corr: Vec<&str> = corr.split(',').collect();
        let rank = |ids: Vec<u32>| ids.iter().position(|&id| corr.contains(&ix.words[id as usize].as_str())).map(|r| r + 1);
        let lev = rank(ix.fuzzy_suggest_ids(m, 10, false));
        let osa = rank(ix.fuzzy_suggest_ids(m, 10, true));
        let in_vocab = ix.contains(m).is_some();
        let completions = ix.suggest_ids(m, 10);
        let comp_rank = completions.iter().position(|&id| corr.contains(&ix.words[id as usize].as_str())).map(|r| r + 1);
        let corr_in_vocab = corr.iter().any(|c| ix.contains(c).is_some());
        rows.push(json!({"m": m, "corr": corr, "corr_in_vocab": corr_in_vocab, "in_vocab": in_vocab,
                         "has_completions": !completions.is_empty(), "completion_rank": comp_rank,
                         "lev_rank": lev, "osa_rank": osa}));
    }
    let v = json!({"n_words": ix.len(), "rows": rows});
    std::fs::write(out, serde_json::to_string(&v).unwrap()).unwrap();
}

fn main() {
    let a = args();
    let (typed, worst) = read_prefixes(&format!("{}/prefixes_{}.json", a.cache, a.n));
    let mut mem = Map::new();
    mem.insert("start".into(), json!(rss_mb("VmRSS:")));
    let vocab = load_vocab_tsv(&format!("{}/vocab_{}.tsv", a.cache, a.n)).unwrap();
    mem.insert("after_vocab".into(), json!(rss_mb("VmRSS:")));
    let t0 = Instant::now();
    let mut ix = Index::new(vocab, a.k);
    let build_s = t0.elapsed().as_secs_f64();
    mem.insert("after_structure".into(), json!(rss_mb("VmRSS:")));
    let t0 = Instant::now();
    ix.build_topk();
    let topk_s = t0.elapsed().as_secs_f64();
    mem.insert("after_topk".into(), json!(rss_mb("VmRSS:")));
    let t0 = Instant::now();
    ix.build_best();
    let best_s = t0.elapsed().as_secs_f64();
    mem.insert("after_best".into(), json!(rss_mb("VmRSS:")));

    if let Some(out) = a.typo {
        typo_eval(&ix, &a.cache, &out);
        return;
    }

    let k = a.k;
    let (empty, _) = sample_calls(|| black_box(1u32), 15, 200_000);
    let mut methods = Map::new();
    {
        let ixr = &ix;
        let f_orig = |p: &str, _k: usize| ixr.complete_all(p);
        let f_topk = |p: &str, k: usize| ixr.suggest(p, k);
        let f_best = |p: &str, k: usize| ixr.suggest_bestfirst(p, k);
        let f_heap = |p: &str, k: usize| ixr.suggest_heap(p, k);
        let f_sa = |p: &str, k: usize| ixr.sorted_suggest(p, k);
        let f_sr = |p: &str, k: usize| ixr.sorted_suggest_ranked(p, k);
        macro_rules! run {
            ($name:expr, $f:expr) => {
                eprintln!("  timing {}", $name);
                methods.insert($name.into(), json!({"by_length": time_method(&$f, k, &typed, a.samples),
                                                     "worst": worst_cases(&$f, k, &worst, a.samples)}));
            };
        }
        run!("original", f_orig);
        run!("topk", f_topk);
        run!("bestfirst", f_best);
        run!("heap", f_heap);
        run!("sorted_alpha", f_sa);
        run!("sorted_ranked", f_sr);
    }
    let mut rec = Map::new();
    rec.insert("language".into(), json!("rust"));
    rec.insert("structure".into(), json!("all"));
    rec.insert("k".into(), json!(k));
    rec.insert("samples".into(), json!(a.samples));
    rec.insert("env".into(), env());
    rec.insert("n_words".into(), json!(ix.len()));
    rec.insert("n_nodes".into(), json!(ix.node_count()));
    rec.insert("build_s".into(), json!(build_s));
    rec.insert("precompute_topk_s".into(), json!(topk_s));
    rec.insert("precompute_best_s".into(), json!(best_s));
    rec.insert("empty_call_us".into(), summarize(&empty));
    rec.insert("methods".into(), Value::Object(methods));
    let mut tr = Map::new();
    for (_, items) in &typed {
        for (p, _) in items {
            tr.insert(p.clone(), json!(ix.suggest(p, k)));
        }
    }
    rec.insert("topk_results".into(), Value::Object(tr));

    if a.full {
        eprintln!("  boundary");
        let ixr = &ix;
        // Tauri-style command: lookup in Rust, return top-k, JSON across the boundary.
        let lookup = |p: &str, k: usize| ixr.suggest_scored(p, k);
        let enc = |p: &str, k: usize| serde_json::to_string(&ixr.suggest_scored(p, k)).unwrap();
        let encdec = |p: &str, k: usize| {
            let s = serde_json::to_string(&ixr.suggest_scored(p, k)).unwrap();
            serde_json::from_str::<Vec<SuggestionOwned>>(&s).unwrap()
        };
        let orig_encdec = |p: &str, _k: usize| {
            let s = serde_json::to_string(&ixr.complete_all(p)).unwrap();
            serde_json::from_str::<Vec<String>>(&s).unwrap()
        };
        let mut payload = Map::new();
        for (l, items) in &typed {
            let xs: Vec<f64> = items.iter().map(|(p, _)| enc(p, k).len() as f64).collect();
            payload.insert(l.clone(), summarize(&xs));
        }
        let mut orig_payload = Map::new();
        for (l, items) in &typed {
            let xs: Vec<f64> = items.iter().map(|(p, _)| serde_json::to_string(&ixr.complete_all(p)).unwrap().len() as f64).collect();
            orig_payload.insert(l.clone(), summarize(&xs));
        }
        rec.insert("boundary".into(), json!({
            "lookup_scored": time_method(&lookup, k, &typed, a.samples),
            "lookup_encode": time_method(&enc, k, &typed, a.samples),
            "lookup_encode_decode": time_method(&encdec, k, &typed, a.samples),
            "worst_lookup_encode_decode": worst_cases(&encdec, k, &worst, a.samples),
            "original_lookup_encode_decode": time_method(&orig_encdec, k, &typed, a.samples),
            "original_worst_lookup_encode_decode": worst_cases(&orig_encdec, k, &worst, a.samples),
            "payload_bytes": payload, "original_payload_bytes": orig_payload}));

        eprintln!("  fuzzy");
        let queries = read_lines(&format!("{}/fuzzy_queries.txt", a.cache));
        let mut lat = Vec::new();
        let mut d1 = Vec::new();
        let mut d2 = Vec::new();
        let mut used_d2 = 0;
        let mut results = Map::new();
        for q in &queries {
            if ix.fuzzy_search(q, 1, false).len() < k {
                used_d2 += 1;
            }
            lat.push(median_of(&sample_calls(|| ix.fuzzy_suggest_ids(q, k, false), 5, 0).0));
            d1.push(median_of(&sample_calls(|| ix.fuzzy_search(q, 1, false), 3, 0).0));
            d2.push(median_of(&sample_calls(|| ix.fuzzy_search(q, 2, false), 3, 0).0));
            results.insert(q.clone(), json!(ix.fuzzy_suggest(q, k)));
        }
        let brute = |q: &str| -> Vec<u32> {
            let qb = q.as_bytes();
            (0..ix.len() as u32)
                .filter(|&id| {
                    let w = ix.words[id as usize].as_bytes();
                    w.len().abs_diff(qb.len()) <= 2 && levenshtein(qb, w) <= 2
                })
                .collect()
        };
        let mut bf = Vec::new();
        for q in queries.iter().take(20) {
            let mut a1 = brute(q);
            let mut b1: Vec<u32> = ix.fuzzy_search(q, 2, false).into_iter().map(|(_, id)| id).collect();
            a1.sort();
            b1.sort();
            assert_eq!(a1, b1, "brute force disagrees on {q}");
            bf.push(median_of(&sample_calls(|| brute(q), 3, 0).0));
        }
        rec.insert("fuzzy".into(), json!({"queries": queries.len(), "needed_distance_2": used_d2,
            "suggest_median_us_per_query": lat, "suggest_summary": summarize(&lat),
            "search_d1_summary": summarize(&d1), "search_d2_summary": summarize(&d2),
            "bruteforce_d2_first20_summary": summarize(&bf), "trie_d2_first20_summary": summarize(&d2[..20.min(d2.len())]),
            "results": results}));

        eprintln!("  learn");
        let picks: Vec<String> = read_lines(&format!("{}/picks.txt", a.cache)).into_iter().filter(|w| ix.contains(w).is_some()).collect();
        let mut times = Vec::with_capacity(picks.len());
        for w in &picks {
            let t0 = Instant::now();
            ix.learn(black_box(w), LEARN_DELTA).unwrap();
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        let learned: Vec<(String, u64)> = ix.words.iter().cloned().zip(ix.weight.iter().cloned()).collect();
        let t0 = Instant::now();
        let mut fresh = Index::new(learned, k);
        let rebuild_trie = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        fresh.build_topk();
        fresh.build_best();
        let rebuild_caches = t0.elapsed().as_secs_f64();
        let same = typed.iter().all(|(_, items)| items.iter().all(|(p, _)| fresh.suggest(p, k) == ix.suggest(p, k)));
        let distinct: std::collections::HashSet<&String> = picks.iter().collect();
        rec.insert("learn".into(), json!({"picks": picks.len(), "distinct": distinct.len(), "delta": LEARN_DELTA,
            "update_us": summarize(&times), "update_us_raw": times,
            "rebuild_trie_s": rebuild_trie, "rebuild_caches_s": rebuild_caches,
            "matches_rebuild_on_typed_prefixes": same}));
    }
    mem.insert("peak".into(), json!(rss_mb("VmHWM:")));
    rec.insert("mem_mb".into(), Value::Object(mem));
    std::fs::write(&a.out, serde_json::to_string(&Value::Object(rec)).unwrap()).unwrap();
    eprintln!("done rust {}", a.n);
}
