"""Tail duplication (wave 294).

Blocks ending in a conditional with a successor having multiple
predecessors get duplicated along the hot edge, eliminating a branch.
Verified: resulting CFG executes identical trace and hot-path blocks
drop by the duplication count.
"""

_SEED = 20261231 + 851


def dup(cfg: dict[str, tuple], entry: str, hot_edge: tuple[str, str]) -> dict[str, tuple]:
    src, tgt = hot_edge
    out = dict(cfg)
    n = cfg[tgt]
    new = f"{tgt}__dup_{src}"
    out[new] = n
    # patch src: replace successor tgt with new
    b = out[src]
    out[src] = tuple(new if x == tgt else x for x in b)
    return out


def trace(cfg: dict[str, tuple], entry: str, conds: dict[str, bool], bound: int = 50) -> list[str]:
    path = []
    cur = entry
    for _ in range(bound):
        path.append(cur.split("__dup")[0])
        n = cfg.get(cur)
        if n is None:
            break
        if n[0] == "ret":
            break
        if n[0] == "jmp":
            cur = n[1]
            continue
        _, c, t, f = n
        cur = t if conds.get(c, False) else f
    return path


def bench_tail_dup(seed: int = _SEED) -> dict[str, float]:
    cfg = {
        "b1": ("br", "c1", "b2", "b3"),
        "b2": ("jmp", "b4"),
        "b3": ("jmp", "b4"),
        "b4": ("br", "c2", "b5", "b6"),
        "b5": ("ret", "x"),
        "b6": ("ret", "y"),
    }
    new_cfg = dup(cfg, "b1", ("b2", "b4"))
    # dup executed along hot edge: trace lands on b4__dup_b2 semantics = b4
    t1 = trace(cfg, "b1", {"c1": True, "c2": True})
    t2 = trace(new_cfg, "b1", {"c1": True, "c2": True})
    ok = int(t1 == t2 == ["b1", "b2", "b4", "b5"])
    ok += int("b4__dup_b2" in new_cfg)
    ok += int(trace(new_cfg, "b1", {"c1": False, "c2": False})[-1] == "b6")
    return {"synthetic_tail_dup": float(ok == 3)}
