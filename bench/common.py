"""Shared helpers for the Python benchmarks: timing, memory, environment."""
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

pc = time.perf_counter_ns


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def summarize(xs):
    s = sorted(xs)
    n = len(s)
    return {"n": n, "min": s[0], "median": statistics.median(s), "mean": statistics.fmean(s),
            "sd": statistics.pstdev(s) if n > 1 else 0.0,
            "p95": s[min(n - 1, int(0.95 * n))], "max": s[-1]}


def sample_calls(f, args, samples=15, target_ns=200_000, max_reps=200_000):
    """Per-call time in microseconds: `samples` warm samples, each the mean of
    `reps` back-to-back calls, with reps chosen so a sample lasts >= target_ns."""
    f(*args)  # warm-up
    reps = 1
    while True:
        t0 = pc()
        for _ in range(reps):
            f(*args)
        el = pc() - t0
        if el >= target_ns or reps >= max_reps:
            break
        reps = min(max_reps, max(reps * 2, int(reps * target_ns / max(el, 1)) + 1))
    out = []
    for _ in range(samples):
        t0 = pc()
        for _ in range(reps):
            f(*args)
        out.append((pc() - t0) / reps / 1000.0)
    return out, reps


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def environment(repo_dir):
    cpu = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    mem = None
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemTotal:"):
            mem = round(int(line.split()[1]) / 1024 / 1024, 2)
    try:
        commit = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    try:
        profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True,
                                 timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        profile = None
    try:
        rustc = subprocess.run(["rustc", "--version"], capture_output=True, text=True,
                               timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        rustc = None
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "mem_total_gib": mem,
            "os": platform.platform(), "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "rustc": rustc,
            "rust_profile": {"opt_level": 3, "lto": True, "codegen_units": 1},
            "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "power_profile": profile, "loadavg_start": os.getloadavg(),
            "repo_commit": commit, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def load_vocab(path):
    words, weights = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            w, c = line.rstrip("\n").split("\t")
            words.append(w)
            weights[w] = int(c)
    return words, weights
