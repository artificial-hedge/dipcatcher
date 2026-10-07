"""Multi-stage scenario tree for a capacity/demand problem (SYNTHETIC).

Binary demand tree (high/low) over 3 stages; backward induction
computes optimal expected profit and the stage-1 decision under
nonanticipativity. Bench reports expected objective vs the perfect-
foresight upper bound and deterministic-mean baseline.
"""

import numpy as np

_SCEN_PROB = 0.5
_DEMAND_HI, _DEMAND_LO = 10.0, 4.0
_BUILD = 3.0
_MARGIN = np.array([5.0, 4.0])


def _stage_profit(x: float, d: float) -> float:
    return float(_MARGIN[0] * min(x, d))


def _expected_opt() -> tuple[float, float, float]:
    # stage 1 build x1; stage 2/3 demand outcomes hi/lo; can build more
    # at stages 2,3 (at same cost) then sell min(capacity, demand)
    leaves = [
        (_DEMAND_HI, _DEMAND_HI),
        (_DEMAND_HI, _DEMAND_LO),
        (_DEMAND_LO, _DEMAND_HI),
        (_DEMAND_LO, _DEMAND_LO),
    ]
    best = -np.inf
    xs = np.linspace(0, 20, 81)
    for x1 in xs:
        tot = 0.0
        for d2, d3 in leaves:
            # stage2: observe d2, may add x2 >= 0
            best23 = -np.inf
            for x2 in xs:
                v2 = _stage_profit(x1 + x2, d2) - _BUILD * x2
                for x3 in xs:
                    v3 = v2 + _stage_profit(x1 + x2 + x3, d3) - _BUILD * x3
                    best23 = max(best23, v3)
            tot += 0.25 * best23
        tot -= _BUILD * x1
        if tot > best:
            best = tot
    # perfect foresight bound
    pf = 0.0
    for d2, d3 in leaves:
        best_pf = -np.inf
        for x in xs:
            v = _stage_profit(x, d2) + _stage_profit(x, d3) - _BUILD * x
            best_pf = max(best_pf, v)
        pf += 0.25 * best_pf
    # expected-value solution: build for mean demand, then recourse
    dm = np.array(leaves).mean(0)
    x_ev = max(xs, key=lambda x: _stage_profit(x, dm[0]) - _BUILD * x)
    det = 0.0
    for d2, d3 in leaves:
        best23 = -np.inf
        for x2 in xs:
            v2 = _stage_profit(x_ev + x2, d2) - _BUILD * x2
            for x3 in xs:
                best23 = max(best23, v2 + _stage_profit(x_ev + x2 + x3, d3) - _BUILD * x3)
        det += 0.25 * best23
    det -= _BUILD * x_ev
    return best, pf, det


def bench_scenario_tree(seed: int = 5503) -> dict[str, float]:
    stoch, pf, det = _expected_opt()
    return {
        "synthetic_tree_obj": stoch,
        "synthetic_tree_pf": pf,
        "synthetic_tree_det": det,
        "synthetic_tree_evpi": pf - stoch,
        "synthetic_tree_vss": stoch - det,
    }
