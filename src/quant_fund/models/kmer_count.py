"""k-mer spectrum: count + distinctness vs oracle (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 732


def kmer_spectrum(seq: str, k: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        out[kmer] = out.get(kmer, 0) + 1
    return out


def bench_kmer_count(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    seq = "".join("ACGT"[rng.randint(4)] for _ in range(200))
    k = 5
    spec = kmer_spectrum(seq, k)
    total = sum(spec.values())
    top = max(spec.values())
    # oracle: most common k-mer found by overlapping brute scan
    windows = [seq[i : i + k] for i in range(len(seq) - k + 1)]
    brute = max(windows.count(w) for w in set(windows))
    ok = float(total == len(seq) - k + 1 and top == brute)
    return {"synthetic_kmer_exact": ok}
