"""Profile-guided basic-block reordering (wave 294) (SYNTHETIC).

Greedy fallthrough chaining: each block's hottest successor is placed
immediately after it → minimizes taken-branch count vs arbitrary
layout oracle.
"""

_SEED = 20261231 + 853


def layout(cfg: dict[str, tuple], entry: str, prof: dict[tuple[str, str], int]) -> list[str]:
    order = []
    placed = set()
    cur: str | None = entry
    while cur:
        order.append(cur)
        placed.add(cur)
        n = cfg[cur]
        succs = (
            [s for s in (n[2], n[3]) if s] if n[0] == "br" else ([n[1]] if n[0] == "jmp" else [])
        )
        succs = [s for s in succs if s not in placed]
        if not succs:
            cur = None
            continue

        def _edge_prof(succ: str, src: str = cur or "") -> int:
            return -prof.get((src, succ), 0)

        succs.sort(key=_edge_prof)
        cur = succs[0]
    for b in cfg:
        if b not in placed:
            order.append(b)
    return order


def taken_branches(
    order: list[str], cfg: dict[str, tuple], prof: dict[tuple[str, str], int] | None = None
) -> int:
    idx = {b: i for i, b in enumerate(order)}
    taken = 0
    for b, n in cfg.items():
        succs = (
            [s for s in (n[2], n[3]) if s] if n[0] == "br" else ([n[1]] if n[0] == "jmp" else [])
        )
        for s in succs:
            if idx[s] != idx[b] + 1:
                taken += (prof or {}).get((b, s), 1)
    return taken


def bench_bb_reorder(seed: int = _SEED) -> dict[str, float]:
    cfg = {
        "e": ("br", "c", "a", "z"),
        "a": ("jmp", "m"),
        "m": ("jmp", "x"),
        "z": ("jmp", "x"),
        "x": ("ret", "r"),
    }
    prof = {("e", "a"): 90, ("e", "z"): 10}
    prof2 = {**prof, ("z", "x"): 10, ("a", "m"): 90, ("m", "x"): 90}
    o = layout(cfg, "e", prof)
    ok = int(o.index("a") == o.index("e") + 1)
    # weighted taken: good layout 20 vs cold-first layout 100
    ok += int(taken_branches(o, cfg, prof2) < taken_branches(["e", "z", "a", "m", "x"], cfg, prof2))
    return {"synthetic_bb_reorder": float(ok == 2)}
