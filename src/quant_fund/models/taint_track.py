"""Dynamic taint tracking: mark source-derived values through ops."""

_SEED = 20261231 + 671


def taint_prop(ops: list[tuple[str, int, int]]) -> set[int]:
    """ops: ('src',id,-) | ('add',a,b)->new | ('kill',id,-). Returns tainted ids."""
    tainted: set[int] = set()
    next_id = 1000
    vals: dict[int, int] = {}
    for op in ops:
        if op[0] == "src":
            tainted.add(op[1])
            vals[op[1]] = 1
        elif op[0] == "add":
            nid = next_id
            next_id += 1
            if op[1] in tainted or op[2] in tainted:
                tainted.add(nid)
            vals[nid] = vals.get(op[1], 0) + vals.get(op[2], 0)
        elif op[0] == "kill":
            tainted.discard(op[1])
    return tainted


def bench_taint_track(seed: int = _SEED) -> dict[str, float]:
    ok = 0.0
    trials = 40
    for _ in range(trials):
        # build a chain: src a; add a,b; kill a; check new id still tainted
        ops = [("src", 1, -1), ("add", 1, 2)]
        t = taint_prop(ops)
        ok += float(1000 in t)
        # kill test
        ops2 = [("src", 1, -1), ("kill", 1, -1), ("add", 1, 2)]
        t2 = taint_prop(ops2)
        ok += float(1000 not in t2)
    return {"synthetic_taint_correct": ok / (2 * trials)}
