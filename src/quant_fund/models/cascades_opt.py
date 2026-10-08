"""Cascades-style memoized optimizer: join enumeration over rule set (SYNTHETIC)."""

import itertools

import numpy as np

_SEED = 20261231 + 600


def _cost(plan: tuple, cards: dict[str, float], join_sel: float = 0.1) -> float:
    """plan: ('scan', t) | ('nlj'|'hj', left, right). Cost model: scan=card,
    nlj = l_cost + l_card*r_card*sel_frac, hj = l+r build+probe."""
    kind = plan[0]
    if kind == "scan":
        return cards[plan[1]]
    _, lp, rp = plan
    lc, rc = _card(lp, cards, join_sel), _card(rp, cards, join_sel)
    base = _cost(lp, cards, join_sel) + _cost(rp, cards, join_sel)
    return base + (lc * rc if kind == "nlj" else lc + rc)


def _card(plan: tuple, cards: dict[str, float], join_sel: float) -> float:
    if plan[0] == "scan":
        return cards[plan[1]]
    return _card(plan[1], cards, join_sel) * _card(plan[2], cards, join_sel) * join_sel


def optimize(tables: list[str], cards: dict[str, float], join_sel: float = 0.1) -> tuple:
    """Exhaustive left-deep + bushy enumeration (memoized)."""
    memo: dict[frozenset, tuple] = {}

    def rec(ts: frozenset) -> tuple:
        if ts in memo:
            return memo[ts]
        if len(ts) == 1:
            memo[ts] = ("scan", next(iter(ts)))
            return memo[ts]
        best, best_plan = np.inf, None
        for k in range(1, len(ts)):
            for sub in itertools.combinations(ts, k):
                L, R = frozenset(sub), ts - frozenset(sub)
                for a, b in [(rec(L), rec(R))]:
                    for op in ("nlj", "hj"):
                        pl = (op, a, b)
                        c = _cost(pl, cards, join_sel)
                        if c < best:
                            best, best_plan = c, pl
        if not (best_plan is not None):
            raise ValueError("best_plan is not None")
        memo[ts] = best_plan
        return memo[ts]

    return rec(frozenset(tables))


def bench_cascades_opt(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(30):
        tables = [f"t{i}" for i in range(4)]
        cards = {t: float(rng.randint(10, 10_000)) for t in tables}
        plan = optimize(tables, cards)
        c = _cost(plan, cards)
        # exhaustive oracle over all bushy plans
        best = np.inf
        for perm in itertools.permutations(tables):
            for op in ("nlj", "hj"):
                pl: tuple = ("scan", perm[0])
                for t in perm[1:]:
                    pl = (op, pl, ("scan", t))
                best = min(best, _cost(pl, cards))
        if c <= best + 1e-9:
            ok += 1
    return {"synthetic_cascades_optimal": ok / 30}
