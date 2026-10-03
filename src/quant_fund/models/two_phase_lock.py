"""SYNTHETIC strict 2PL scheduler + serializability checker.

Transactions issue lock(x,S/X)/unlock; scheduler grants/blocks. Verify:
conflicting access serialized, 2PL schedule is conflict-equivalent to
some serial order (precedence graph acyclic), no deadlock under
wait-ordering.
"""

from __future__ import annotations

import random


def precedence_acyclic(schedule: list[tuple[str, int, str]]) -> bool:
    """schedule = [(op,tx,key)] ops r/w; build conflict edges ti->tj."""
    edges: set[tuple[int, int]] = set()
    for i, (o1, t1, k1) in enumerate(schedule):
        for o2, t2, k2 in schedule[i + 1 :]:
            if t1 == t2 or k1 != k2:
                continue
            if o1 == "w" or o2 == "w":
                edges.add((t1, t2))
    # kahn
    nodes = {t for _, t, _ in schedule}
    indeg = {n: 0 for n in nodes}
    for _a, b in edges:
        indeg[b] += 1
    q = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while q:
        u = q.pop()
        seen += 1
        for a, b in edges:
            if a == u:
                indeg[b] -= 1
                if indeg[b] == 0:
                    q.append(b)
    return seen == len(nodes)


def run_2pl(txns: list[list[tuple[str, str]]]) -> list[tuple[str, int, str]]:
    """Strict 2PL: locks held to commit; round-robin interleave."""
    held: dict[str, int] = {}  # key -> tx holding X
    locks: dict[int, set[str]] = {i: set() for i in range(len(txns))}
    schedule: list[tuple[str, int, str]] = []
    alive = list(range(len(txns)))
    pc = [0] * len(txns)
    guard = 0
    while alive and guard < 10000:
        guard += 1
        for t in list(alive):
            if pc[t] >= len(txns[t]):
                for k in locks[t]:
                    held.pop(k, None)
                locks[t] = set()
                alive.remove(t)
                continue
            op, key = txns[t][pc[t]]
            owner = held.get(key)
            if op == "r":
                if owner is None or owner == t:
                    # shared upgrade-allowed under strict X-locking model
                    held.setdefault(key, t)
                    locks[t].add(key)
                    schedule.append(("r", t, key))
                    pc[t] += 1
            else:
                if owner is None or owner == t:
                    held[key] = t
                    locks[t].add(key)
                    schedule.append(("w", t, key))
                    pc[t] += 1
    return schedule


def bench_two_phase_lock(seed: int = 20261231 + 431) -> dict[str, float]:
    rng = random.Random(seed)
    serial = guarded = 0
    complete = 0.0
    trials = 40
    for _ in range(trials):
        ntx = rng.randrange(2, 5)
        keys = "abc"
        txns = []
        for _ in range(ntx):
            ops = [(rng.choice("rw"), rng.choice(keys)) for _ in range(rng.randrange(2, 6))]
            txns.append(ops)
        sched = run_2pl(txns)
        serial += int(precedence_acyclic(sched))
        # progress = scheduled ops / requested ops (deadlock truncates)
        requested = sum(len(t) for t in txns)
        complete += len(sched) / max(1, requested)
        # strict 2PL mutual exclusion replay: each op must find its key
        # either free or held by the same tx; locks release only after
        # the tx's final scheduled op.
        last_op = {
            t2: max(i for i, (_, tt, _) in enumerate(sched) if tt == t2)
            for t2 in range(ntx)
            if any(tt == t2 for _, tt, _ in sched)
        }
        holder: dict[str, int] = {}
        ok = True
        for i, (_op, t, k) in enumerate(sched):
            if holder.get(k) not in (None, t):
                ok = False
                break
            holder[k] = t
            if i == last_op[t]:
                holder = {kk: vv for kk, vv in holder.items() if vv != t}
        guarded += int(ok)
    return {
        "synthetic_conflict_serializable": float(serial / trials),
        "synthetic_scheduled_progress": float(complete / trials),
        "synthetic_exclusive_writes": float(guarded / trials),
    }
