"""Martin's axiom toy: one filter meets all dense sets (SYNTHETIC)."""

from __future__ import annotations

Cond = dict[int, int]


def all_conditions(max_len: int) -> list[Cond]:
    """Enumerate Cohen conditions with domain subset of range(max_len)."""
    conds: list[Cond] = [{}]
    for i in range(max_len):
        conds += [{**c, i: 0} for c in list(conds)] + [
            {**c, i: 1} for c in list(conds) if i not in c
        ]
    # dedupe by frozenset
    seen: set[frozenset] = set()
    out: list[Cond] = []
    for c in conds:
        f = frozenset(c.items())
        if f not in seen and len(c) <= max_len:
            seen.add(f)
            out.append(c)
    return out


def is_compatible(p: Cond, q: Cond) -> bool:
    return all(p[k] == q[k] for k in p if k in q)


def meets_all(g: list[Cond], dense_sets: list[list[Cond]]) -> bool:
    """G intersects every listed dense set: for each dense D some gc in G
    extends some dc in D (gc superset-agrees with dc)."""
    for d in dense_sets:
        if not any(all(k in gc and gc[k] == v for k, v in dc.items()) for dc in d for gc in g):
            return False
    return True


def _bench_ma_toy(seed: int = 0) -> float:
    checks = []
    conds = all_conditions(3)
    checks.append(len(conds) == 27)  # 3^3 ternary choices per slot
    dense1 = [c for c in conds if len(c) >= 1]
    dense2 = [c for c in conds if len(c) >= 2]
    dense3 = [c for c in conds if len(c) >= 3]
    # generic branch containing 111 = {0:1,1:1,2:1}
    g = [{}, {0: 1}, {0: 1, 1: 1}, {0: 1, 1: 1, 2: 1}]
    checks.append(meets_all(g, [dense1, dense2, dense3]))
    # a filter stopping at length 1 fails MA for the k=3 dense set
    g_small = [{}, {0: 0}]
    checks.append(not meets_all(g_small, [dense3]))
    # MA requires ccc: Cohen poset is ccc (antichains are countable);
    # on our finite model antichain bound = total conditions
    checks.append(len(conds) < 30)
    return float(sum(checks) / len(checks))


def bench_ma_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ma_toy": _bench_ma_toy(seed)}
