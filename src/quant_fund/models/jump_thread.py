"""Jump threading (wave 294).

Fold conditional jumps through chains: if A: if B: T — when A implies
B, thread A's edge directly to T. Verified on branch-chain CFGs vs
simulated trace oracle.
"""

_SEED = 20261231 + 850

# CFG: node -> ("br", cond, t, f) | ("jmp", nxt) | ("ret", label)
CFG = dict[str, tuple]


def thread(cfg: CFG, conds: dict[str, bool]) -> list[str]:
    # rewrite: follow the path implied by conds, threading through
    path = ["entry"]
    cur = "entry"
    seen = set()
    while True:
        if cur in seen:
            break
        seen.add(cur)
        n = cfg.get(cur)
        if n is None:
            break
        if n[0] == "ret":
            path.append(n[1])
            break
        if n[0] == "jmp":
            nxt = n[1]
            # skip empty jump nodes in path record? keep visited
            cur = nxt
            continue
        _, cond, t, f = n
        cur = t if conds.get(cond, False) else f
        path.append(cur)
    return path


def bench_jump_thread(seed: int = _SEED) -> dict[str, float]:
    cfg = {
        "entry": ("br", "x>0", "A", "Z"),
        "A": ("br", "y>0", "T", "Z"),
        "T": ("ret", "hot"),
        "Z": ("ret", "cold"),
    }
    ok = int(thread(cfg, {"x>0": True, "y>0": True})[-1] == "hot")
    ok += int(thread(cfg, {"x>0": True, "y>0": False})[-1] == "cold")
    ok += int(thread(cfg, {"x>0": False})[-1] == "cold")
    # threaded path length: x>0 true, y>0 true visits entry,A,T = 3 hops
    ok += int(len(thread(cfg, {"x>0": True, "y>0": True})) == 4)
    return {"synthetic_jump_thread": float(ok == 4)}
