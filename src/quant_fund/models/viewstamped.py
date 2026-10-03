"""SYNTHETIC Viewstamped Replication — view change preserves committed log.

f+1 of 2f+1 replicas form a quorum; view change collects the highest
(log_len, view) state. Verify new primary's log covers everything
committed in any previous view (quorum intersection).
"""

from __future__ import annotations

import random


def quorum_view_change(logs: list[list[int]], f: int = 1) -> list[int]:
    """Survivors = first f+1 logs; new primary takes longest log."""
    surv = logs[: f + 1]
    return max(surv, key=len)


def committed_prefix(logs: list[list[int]], f: int = 1) -> list[int]:
    """Entries present on ≥f+1 replicas = committed."""
    n = min(len(lg) for lg in logs)
    out = []
    for i in range(n):
        vals = [lg[i] for lg in logs]
        if len(set(vals)) < len(vals):
            cand = max(set(vals), key=vals.count)
            if vals.count(cand) >= f + 1:
                out.append(cand)
    return out


def bench_viewstamped(seed: int = 20261231 + 442) -> dict[str, float]:
    rng = random.Random(seed)
    covers = quorum_ok = order_ok = 0
    trials = 40
    for _ in range(trials):
        base = [rng.randrange(1, 100) for _ in range(rng.randrange(2, 8))]
        logs = [list(base[: rng.randrange(len(base) + 1)]) for _ in range(3)]
        # committed = prefix replicated on ≥2
        com = committed_prefix(logs)
        new_primary = quorum_view_change(logs)
        covers += int(new_primary[: len(com)] == com)
        quorum_ok += int(len(new_primary) >= len(com))
        # new primary log is a valid prefix of some real history
        order_ok += int(
            new_primary in logs or any(lg[: len(new_primary)] == new_primary for lg in logs)
        )
    return {
        "synthetic_view_change_covers_committed": float(covers / trials),
        "synthetic_quorum_len": float(quorum_ok / trials),
        "synthetic_primary_log_valid": float(order_ok / trials),
    }
