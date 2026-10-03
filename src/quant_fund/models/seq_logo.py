"""Sequence logo: per-column information content of an alignment."""

import numpy as np

_SEED = 20261231 + 733
_ALPH = "ACGT"


def column_ic(alignment: list[str], j: int, base: float = 2.0) -> float:
    counts = np.zeros(4)
    for s in alignment:
        counts[_ALPH.index(s[j])] += 1
    p = counts / counts.sum()
    ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
    return float(np.log2(4) - ent)


def bench_seq_logo(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n, L = 30, 8
    alignment = []
    for _ in range(n):
        s = []
        for j in range(L):
            # column 0 fully conserved, others noisy
            if j == 0:
                s.append("A")
            else:
                s.append("ACGT"[rng.randint(4)])
        alignment.append("".join(s))
    ics = [column_ic(alignment, j) for j in range(L)]
    ok = float(ics[0] == 2.0 and max(ics[1:]) < 1.0)
    return {"synthetic_logo_ic": ok}
