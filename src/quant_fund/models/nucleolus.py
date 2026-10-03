"""Nucleolus (Schmeidler 1969) — lexicographic minimization of the
sorted excess vector via successive LPs. Also hosts the shared
coalitional-game spec + classic factories (glove, airport, voting)
used across the wave-120 cooperative-game modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

FloatArray = NDArray[np.float64]


@dataclass
class CoopGame:
    """TU game: n players, v(S) for every coalition bitmask S."""

    n: int
    values: FloatArray  # length 2**n, index = bitmask

    def v(self, s: int) -> float:
        return float(self.values[s])


def glove_game(l_gloves: int = 2, r_gloves: int = 1) -> CoopGame:
    """Left-glove owners pair with right-glove owners: v(S) = min
    (#left in S, #right in S). Players 0..l-1 left, l..l+r-1 right."""
    n = l_gloves + r_gloves
    vals = np.zeros(2**n)
    for s in range(2**n):
        nl = sum(1 for i in range(l_gloves) if s & (1 << i))
        nr = sum(1 for i in range(l_gloves, n) if s & (1 << i))
        vals[s] = min(nl, nr)
    return CoopGame(n, vals)


def airport_game(costs: FloatArray) -> CoopGame:
    """Airport cost game: v(S) = c(S) = max cost of members (cost
    game convention — worth = −cost for value form: use savings =
    sum of max-cost savings). Implemented as cost game v(S)=max c_i."""
    n = len(costs)
    vals = np.zeros(2**n)
    for s in range(2**n):
        members = [costs[i] for i in range(n) if s & (1 << i)]
        vals[s] = max(members) if members else 0.0
    return CoopGame(n, vals)


def voting_game(weights: FloatArray, quota: float) -> CoopGame:
    """Simple majority game: v(S) = 1 if Σ_{i∈S} w_i ≥ quota."""
    n = len(weights)
    vals = np.zeros(2**n)
    for s in range(2**n):
        tot = sum(weights[i] for i in range(n) if s & (1 << i))
        vals[s] = float(tot >= quota)
    return CoopGame(n, vals)


def excesses(g: CoopGame, x: FloatArray) -> FloatArray:
    """Excess vector e(S,x) = v(S) − x(S) over all proper nonempty
    coalitions."""
    n = g.n
    out = []
    for s in range(1, 2**n):
        if s == 2**n - 1:
            continue
        xs = sum(x[i] for i in range(n) if s & (1 << i))
        out.append(g.v(s) - xs)
    arr: FloatArray = np.array(out)
    return arr


def nucleolus(g: CoopGame, max_lps: int = 8) -> FloatArray:
    """Nucleolus via successive LP minimization of the largest
    excess, freezing tight coalitions each round."""
    n = g.n
    x = np.full(n, g.v(2**n - 1) / n)
    fixed: set[int] = set()
    for _ in range(max_lps):
        # LP: min eps s.t. v(S) − x(S) ≤ eps ∀S∉fixed
        #         x(S) = v(S) − eps_S* ∀S∈fixed
        #         x(N) = v(N), x_i ≥ v({i})
        free_coals = [s for s in range(1, 2**n - 1) if s not in fixed]
        m = len(free_coals)
        c = np.zeros(n + 1)
        c[n] = 1.0
        a_ub = np.zeros((m, n + 1))
        b_ub = np.zeros(m)
        for j, s in enumerate(free_coals):
            for i in range(n):
                if s & (1 << i):
                    a_ub[j, i] = -1.0
            a_ub[j, n] = -1.0
            b_ub[j] = -g.v(s)
        a_eq = np.zeros((1 + len(fixed), n + 1))
        b_eq = np.zeros(1 + len(fixed))
        a_eq[0, :n] = 1.0
        b_eq[0] = g.v(2**n - 1)
        for k, s in enumerate(sorted(fixed)):
            for i in range(n):
                if s & (1 << i):
                    a_eq[1 + k, i] = 1.0
            # tight coalition: x(S) = v(S) − max excess at freeze
            # store the excess at freeze time in b_eq via e(S,x*)
            b_eq[1 + k] = g.v(s) - _frozen_excess(g, s)
        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=[(float(g.v(1 << i)), None) for i in range(n)] + [(None, None)],
            method="highs",
        )
        if res.x is None:
            break
        x = res.x[:n]
        eps_new = res.x[n]
        # freeze coalitions whose excess equals eps_new (within tol)
        new_fixed = set()
        for s in free_coals:
            xs = sum(x[i] for i in range(n) if s & (1 << i))
            if abs(g.v(s) - xs - eps_new) < 1e-6:
                new_fixed.add(s)
        for s in new_fixed:
            fixed.add(s)
            _FROZEN[s] = eps_new
        if not new_fixed:
            break
    out: FloatArray = np.asarray(x)
    return out


_FROZEN: dict[int, float] = {}


def _frozen_excess(g: CoopGame, s: int) -> float:  # noqa: ARG001
    return _FROZEN.get(s, 0.0)


def bench_nucleolus(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: nucleolus on the 3-player glove game — the
    right-glove owner (pivot) captures the whole surplus (analytic
    nucleolus = (0,0,1) up to player order), and on the symmetric
    majority game it splits evenly."""
    del seed
    out: dict[str, float] = {}
    g = glove_game(2, 1)
    _FROZEN.clear()
    x = nucleolus(g)
    out["synthetic_nucleolus_efficient"] = float(abs(x.sum() - g.v(7)) < 1e-6)
    # right owner (index 2) should get the bulk of the surplus
    out["synthetic_nucleolus_pivot_share"] = float(x[2])
    v = voting_game(np.array([1.0, 1.0, 1.0]), 2.0)
    _FROZEN.clear()
    xs = nucleolus(v)
    out["synthetic_nucleolus_symmetric_err"] = float(np.abs(xs - 1.0 / 3).max())
    out["synthetic_nucleolus_ok"] = float(
        out["synthetic_nucleolus_efficient"] > 0.5
        and x[2] > 0.5
        and out["synthetic_nucleolus_symmetric_err"] < 0.05
    )
    return out


if __name__ == "__main__":
    print(bench_nucleolus())
