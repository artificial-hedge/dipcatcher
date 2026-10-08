"""Partial redundancy elimination on a tiny CFG (lazy-code-motion style) (SYNTHETIC).

Blocks: {"succ": [...], "exprs": [("e", "x+y"), ...], "kill": [vars]}
An expression is redundant if already evaluated on all paths to a use.
PRE inserts the computation in predecessors where it's missing, then
removes fully-anticipated recomputations.
"""

from __future__ import annotations

_SEED = 20261231 + 901

CFG = dict[str, dict]


def _gen_kill(cfg: CFG, expr: str) -> tuple[dict[str, bool], dict[str, bool]]:
    gen: dict[str, bool] = {}
    kill: dict[str, bool] = {}
    for b, node in cfg.items():
        gen[b] = any(e == expr for e, _ in node["exprs"])
        kill[b] = any(v in expr for v in node.get("kill", ()))
    return gen, kill


def anticipatable(cfg: CFG, entry: str, expr: str) -> set[str]:
    """Backward analysis: expr anticipated at block exit (out-set)."""
    gen, kill = _gen_kill(cfg, expr)
    ant_out: dict[str, bool] = {b: False for b in cfg}
    changed = True
    while changed:
        changed = False
        for b, node in cfg.items():
            # ANT.out[b] = AND over successors of ANT.in[s];
            # ANT.in[s] = gen[s] or (ANT.out[s] and not kill[s])
            succ_in = bool(node["succ"]) and all(
                (gen[s] or (ant_out[s] and not kill[s])) for s in node["succ"]
            )
            if succ_in != ant_out[b]:
                ant_out[b] = succ_in
                changed = True
    return {b for b in cfg if ant_out[b]}


def pre_insert(cfg: CFG, entry: str, expr: str) -> CFG:
    """PRE-lite edge insertion: for each block evaluating `expr`, insert it
    in every predecessor that neither evaluates nor kills it."""
    out: CFG = {
        b: {"succ": list(n["succ"]), "exprs": list(n["exprs"]), "kill": list(n.get("kill", ()))}
        for b, n in cfg.items()
    }
    for b, node in out.items():
        if not any(e == expr for e, _ in node["exprs"]):
            continue
        for pnode in out.values():
            if b not in pnode["succ"]:
                continue
            has = any(e == expr for e, _ in pnode["exprs"])
            kills = any(v in expr for v in pnode["kill"])
            if not has and not kills:
                pnode["exprs"].append((expr, expr))
    return out


def count_evals(cfg: CFG, expr: str) -> int:
    return sum(1 for n in cfg.values() for e, _ in n["exprs"] if e == expr)


def bench_gvn_pre(seed: int = _SEED) -> dict[str, float]:
    # diamond: entry -> {L computes x+y, R doesn't} -> join uses x+y
    cfg: CFG = {
        "entry": {"succ": ["L", "R"], "exprs": [], "kill": []},
        "L": {"succ": ["join"], "exprs": [("x+y", "t")], "kill": []},
        "R": {"succ": ["join"], "exprs": [], "kill": []},
        "join": {"succ": ["use"], "exprs": [("x+y", "u")], "kill": []},
        "use": {"succ": [], "exprs": [], "kill": []},
    }
    score = 0.0
    ant = anticipatable(cfg, "entry", "x+y")
    score += 1.0 if ant == {"entry", "L", "R"} else 0.0
    out = pre_insert(cfg, "entry", "x+y")
    # R must now compute x+y (join anticipates on all paths), and the
    # insertion also hoists to entry (both preds of join must provide it)
    score += 1.0 if any(e == "x+y" for e, _ in out["R"]["exprs"]) else 0.0
    score += 1.0 if any(e == "x+y" for e, _ in out["entry"]["exprs"]) else 0.0
    # insertion phase leaves 4 evals (entry, L, R, join); a later pass reuses
    score += 1.0 if count_evals(out, "x+y") == 4 else 0.0
    # kill blocks anticipation: recompute needed when x killed on a path
    cfg2: CFG = {
        "entry": {"succ": ["L", "R"], "exprs": [], "kill": []},
        "L": {"succ": ["join"], "exprs": [("x+y", "t")], "kill": []},
        "R": {"succ": ["join"], "exprs": [], "kill": ["x"]},
        "join": {"succ": [], "exprs": [("x+y", "u")], "kill": []},
    }
    ant2 = anticipatable(cfg2, "entry", "x+y")
    score += 1.0 if "entry" not in ant2 else 0.0
    return {"synthetic_gvn_pre": score / 5.0}
