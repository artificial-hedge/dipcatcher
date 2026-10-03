"""Generic retrograde WDL analysis on a finite game graph.

Input: succ(s) -> list of successor states, terminal(s) -> +1/-1 mover
outcome or None. The standard fixpoint: a state is won when some
successor is lost for the mover, lost when every successor is won for
the mover; anything unresolved at fixpoint is a draw. Verified against
exhaustive minimax unrolling (cycles -> draw).
"""

import numpy as np

_SEED = 20261231 + 879


def retrograde(succ: dict[int, list[int]], terminal: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = dict(terminal)
    states = list(succ.keys())
    changed = True
    while changed:
        changed = False
        for s in states:
            if s in out:
                continue
            vals = [out[x] for x in succ[s] if x in out]
            if -1 in vals:
                out[s] = 1
                changed = True
            elif len(vals) == len(succ[s]) and all(v == 1 for v in vals):
                out[s] = -1
                changed = True
    for s in states:
        out.setdefault(s, 0)
    return out


def _minimax(
    s: int, succ: dict[int, list[int]], terminal: dict[int, int], depth: int, memo: dict[int, int]
) -> int:
    if s in terminal:
        return terminal[s]
    if depth <= 0:
        return 0
    if s in memo:
        return memo[s]
    vals = [-_minimax(x, succ, terminal, depth - 1, memo) for x in succ[s]]
    memo[s] = max(vals)
    return memo[s]


def bench_retrograde_wdl(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: fixpoint verdicts == deep minimax on acyclic + cyclic graphs."""
    rng = np.random.default_rng(seed)
    n = 200
    # Acyclic DAG: edges only to higher ids.
    succ: dict[int, list[int]] = {}
    terminal: dict[int, int] = {}
    for s in range(n):
        nxt = (
            [int(x) for x in rng.choice(np.arange(s + 1, n), min(2, n - 1 - s), replace=False)]
            if s < n - 1
            else []
        )
        succ[s] = nxt
        if not nxt:
            terminal[s] = int(rng.choice([-1, 1]))
    wd = retrograde(succ, terminal)
    ok1 = all(wd[s] == _minimax(s, succ, terminal, 64, {}) for s in range(n))
    # Cyclic graph: add back-edges so some nodes can only draw.
    succ2 = {k: list(v) for k, v in succ.items()}
    succ2[50].append(10)
    succ2[60].append(30)
    wd2 = retrograde(succ2, terminal)
    mm = {s: _minimax(s, succ2, terminal, 200, {}) for s in range(n)}
    ok2 = all(wd2[s] == mm[s] for s in range(n))
    return {"synthetic_retrograde_wdl": 1.0 if (ok1 and ok2) else 0.0}
