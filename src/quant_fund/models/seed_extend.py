"""Seed-and-extend local alignment (wave 284) (SYNTHETIC).

Find exact k-mer seeds between query and subject, extend each seed
ungapped in both directions while matches dominate, and report the best
scoring hit — a BLAT-style two-hit heuristic.
"""

import numpy as np

_SEED = 20261231 + 790


def seed_extend(query: str, subject: str, k: int = 4) -> tuple[int, int, int]:
    # index subject k-mers
    index: dict[str, list[int]] = {}
    for j in range(len(subject) - k + 1):
        index.setdefault(subject[j : j + k], []).append(j)
    best = (0, 0, 0)  # score, q_start, s_start
    for i in range(len(query) - k + 1):
        for j in index.get(query[i : i + k], []):
            # extend left
            lo_q, lo_s = i, j
            while lo_q > 0 and lo_s > 0 and query[lo_q - 1] == subject[lo_s - 1]:
                lo_q -= 1
                lo_s -= 1
            hi_q, hi_s = i + k, j + k
            while hi_q < len(query) and hi_s < len(subject) and query[hi_q] == subject[hi_s]:
                hi_q += 1
                hi_s += 1
            score = hi_q - lo_q
            if score > best[0]:
                best = (score, lo_q, lo_s)
    return best


def bench_seed_extend(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    alpha = "ACGT"
    subject = "".join(rng.choice(list(alpha), 120))
    # plant a 30-base fragment at subject[50:80] into query at [10:40]
    frag = subject[50:80]
    query = "".join(rng.choice(list(alpha), 60))
    query = query[:10] + frag + query[40:]
    score, qs, ss = seed_extend(query, subject)
    ok = int(score >= 30 and qs <= 10 and qs + score >= 40 and ss <= 50 and ss + score >= 80)
    return {"synthetic_seed_extend": float(ok)}
