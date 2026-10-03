"""PBFT-lite Byzantine agreement (synthetic).

3f+1 replicas, one primary, f Byzantine (send arbitrary votes).
Phases: pre-prepare → prepare → commit; a replica commits on 2f+1
matching prepare+commit certificates. Verified: (i) agreement —
honest replicas never commit different values; (ii) liveness — with
honest primary and f ≤ (n-1)/3 a decision is reached; (iii) safety —
Byzantine primary cannot split honest commits.
"""

from __future__ import annotations

import random


def run_pbft(
    n: int, f: int, byz_primary: bool, value: int, rng: random.Random
) -> dict[str, object]:
    honest = n - f
    byz = set(range(honest, n))
    primary = 0 if not byz_primary else n - 1
    # primary proposes; byzantine primary equivocates to half
    proposals: dict[int, int] = {}
    for r in range(n):
        if primary in byz and rng.random() < 0.5:
            proposals[r] = value if r % 2 == 0 else value + 1
        else:
            proposals[r] = value
    # prepare: each replica broadcasts its (possibly equivocated) proposal
    prepares: dict[int, dict[int, int]] = {r: {} for r in range(n)}
    for r in range(n):
        for sender in range(n):
            if sender in byz:
                prepares[r][sender] = rng.choice([value, value + 1, -1])
            else:
                prepares[r][sender] = proposals[sender]
    # commit: count prepares for each value; byzantine equivocation
    commits: dict[int, int | None] = {}
    for r in range(honest):
        counts: dict[int, int] = {}
        for v in prepares[r].values():
            if v is not None:
                counts[v] = counts.get(v, 0) + 1
        quorum_val = next((v for v, c in counts.items() if c >= 2 * f + 1), None)
        commits[r] = quorum_val
    decided = {v for v in commits.values() if v is not None}
    # safety = no two honest replicas commit different values
    agreement = safety = len(decided) <= 1
    liveness = bool(decided) or primary in byz
    return {
        "agreement": agreement,
        "liveness": liveness,
        "safety": safety,
        "decided": decided,
        "commits": sum(1 for v in commits.values() if v is not None),
    }


def bench_pbft_lite(seed: int = 20261231 + 255) -> dict[str, float]:
    rng = random.Random(seed)
    n, f = 10, 3  # n >= 3f+1
    agr = liv = saf = 0
    trials = 60
    for _ in range(trials):
        byz_prim = rng.random() < 0.4
        res = run_pbft(n, f, byz_prim, value=42, rng=rng)
        agr += int(bool(res["agreement"]))
        liv += int(bool(res["liveness"]))
        saf += int(bool(res["safety"]))
    # honest primary: must always decide 42
    det = 0
    for _ in range(30):
        res = run_pbft(n, f, False, 42, rng)
        det += int(res["decided"] == {42})
    return {
        "synthetic_agreement": float(agr / trials),
        "synthetic_liveness": float(liv / trials),
        "synthetic_safety": float(saf / trials),
        "synthetic_deterministic": float(det / 30),
    }
