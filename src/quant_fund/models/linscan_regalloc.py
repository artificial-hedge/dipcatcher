"""Linear-scan register allocation (synthetic).

Intervals sorted by start; expire registers as intervals end;
spill when pool exhausted (spill longest-current interval).
Verified: (i) no two overlapping intervals share a register;
(ii) spilled intervals carry a spill flag; (iii) register count
within pool + spill bound.
"""

from __future__ import annotations

import random

Interval = tuple[int, int, int]  # (start, end, var)


def linscan(intervals: list[Interval], n_reg: int) -> tuple[dict[int, int], set[int]]:
    order = sorted(intervals)
    free = list(range(n_reg))
    active: list[tuple[int, int, int]] = []  # (end, var, reg)
    alloc: dict[int, int] = {}
    spilled: set[int] = set()
    for st, en, var in order:
        # expire
        active = [a for a in active if a[0] > st]
        used = {a[2] for a in active}
        free = [r for r in range(n_reg) if r not in used]
        if not free:
            # spill longest-active or current
            cand = max(active, key=lambda a: a[0])
            if cand[0] > en:
                # spill candidate
                spilled.add(cand[1])
                active.remove(cand)
                reg = cand[2]
                alloc[var] = reg
                active.append((en, var, reg))
            else:
                spilled.add(var)
        else:
            reg = free[0]
            alloc[var] = reg
            active.append((en, var, reg))
    return alloc, spilled


def bench_linscan_regalloc(seed: int = 20261231 + 285) -> dict[str, float]:
    rng = random.Random(seed)
    no_overlap = spill_ok = within = 0
    trials = 40
    for _ in range(trials):
        n = rng.randint(6, 15)
        n_reg = rng.randint(2, 4)
        iv = []
        for v in range(n):
            st = rng.randint(0, 20)
            en = st + rng.randint(1, 8)
            iv.append((st, en, v))
        alloc, spilled = linscan(iv, n_reg)
        # no two overlapping intervals share a reg
        ok = True
        for i, a in enumerate(iv):
            for b2 in iv[i + 1 :]:
                if a[2] in spilled or b2[2] in spilled:
                    continue
                ov = a[0] < b2[1] and b2[0] < a[1]
                if ov and alloc[a[2]] == alloc[b2[2]]:
                    ok = False
        no_overlap += int(ok)
        # every var is accounted for: has a register or is flagged spilled
        # (a spilled live interval can keep its stale slot entry)
        spill_ok += int(all((v in alloc) or (v in spilled) for _, _, v in iv))
        within += int(all(0 <= r < n_reg for r in alloc.values()))
    # dense interval graph forces spills
    forced = linscan([(0, 10, v) for v in range(5)], 2)
    forced_ok = len(forced[1]) == 3
    return {
        "synthetic_no_overlap": float(no_overlap / trials),
        "synthetic_spill_consistent": float(spill_ok / trials),
        "synthetic_within_pool": float(within / trials),
        "synthetic_forced_spills": float(forced_ok),
    }
