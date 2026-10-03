"""Star multiple-sequence alignment: align each seq to the center seq."""

import numpy as np

_SEED = 20261231 + 729


def global_align(a: str, b: str, match: int = 1, gap: int = -1) -> int:
    """Needleman-Wunsch score (no mismatch penalty for simplicity)."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1))
    dp[:, 0] = np.arange(n + 1) * gap
    dp[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = max(
                dp[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else -1),
                dp[i - 1, j] + gap,
                dp[i, j - 1] + gap,
            )
    return int(dp[n, m])


def star_consensus_score(seqs: list[str]) -> float:
    """Sum of pairwise scores against the median-length center."""
    center = min(seqs, key=lambda s: abs(len(s) - int(np.median([len(x) for x in seqs]))))
    scores = [global_align(center, s) for s in seqs if s is not center]
    return float(np.mean(scores))


def bench_star_msa(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ref = "ACGTACGTACGT"
    seqs = []
    for _ in range(6):
        s = "".join(c if rng.rand() < 0.85 else "ACGT"[rng.randint(4)] for c in ref)
        seqs.append(s)
    related = star_consensus_score(seqs)
    unrelated = star_consensus_score(
        [ref] + ["".join("ACGT"[rng.randint(4)] for _ in range(12)) for _ in range(5)]
    )
    return {"synthetic_star_sep": float(related > unrelated)}
