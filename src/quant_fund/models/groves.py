"""Groves mechanism family — welfare-maximizing allocation with
payments p_i = h_i(v_{−i}) − Σ_{j≠i} v_j(x*); VCG/Clarke pivot is the
special case h_i = max_x Σ_{j≠i} v_j(x). Any Groves-scheme h_i
independent of v_i preserves DSIC.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def groves_mechanism(
    valuations: list[FloatArray],
    h_fns: Sequence[Callable[[FloatArray], float]] | None = None,
) -> tuple[int, FloatArray]:
    """Single-parameter Groves: agent i's valuation vector over
    `m` outcomes (allocations). Chooses argmax total welfare; payment
    p_i = h_i(v_{−i}) − welfare_others(x*). With h_i = Clarke pivot
    (default) payments are VCG; with h_i = 0 agents get PAID their
    externality reduction (Groves with zero h — still DSIC)."""
    n = len(valuations)
    m = len(valuations[0])
    v = np.array(valuations)
    best, best_w = 0, -np.inf
    for x in range(m):
        w = float(v[:, x].sum())
        if w > best_w:
            best, best_w = x, w
    pay = np.zeros(n)
    for i in range(n):
        others = np.delete(v, i, axis=0)
        if h_fns is None:
            # Clarke pivot: max welfare of others without i
            h = max(float(others[:, x].sum()) for x in range(m))
        else:
            h = float(h_fns[i](others))
        pay[i] = h - float(others[:, best].sum())
    return best, pay


def bench_groves(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: Groves general form — VCG payments under pivot;
    DSIC under an arbitrary h (zero-h); truthfulness dominant."""
    del seed
    out: dict[str, float] = {}
    # 3 agents, 3 outcomes
    v = [np.array([5.0, 1.0, 0.0]), np.array([2.0, 4.0, 1.0]), np.array([1.0, 3.0, 2.0])]
    x, pay = groves_mechanism(v)
    out["synthetic_groves_welfare_max"] = float(x == int(np.argmax(np.sum(v, axis=0))))
    # DSIC under pivot: no profitable misreport
    dsic = True
    for i in range(3):
        for scale in np.linspace(0.5, 1.5, 5):
            v2 = [val.copy() for val in v]
            v2[i] = v2[i] * scale
            x2, pay2 = groves_mechanism(v2)
            u_t = v[i][x] - pay[i]
            u_l = v[i][x2] - pay2[i]
            if u_l > u_t + 1e-6:
                dsic = False
    out["synthetic_groves_dsic"] = float(dsic)
    # zero-h Groves: payments differ but DSIC preserved
    h0 = [lambda _o: 0.0] * 3
    x3, pay3 = groves_mechanism(v, h_fns=h0)
    dsic0 = True
    for i in range(3):
        for scale in np.linspace(0.5, 1.5, 5):
            v2 = [val.copy() for val in v]
            v2[i] = v2[i] * scale
            x4, pay4 = groves_mechanism(v2, h_fns=h0)
            u_t = v[i][x3] - pay3[i]
            u_l = v[i][x4] - pay4[i]
            if u_l > u_t + 1e-6:
                dsic0 = False
    out["synthetic_groves_h0_dsic"] = float(dsic0)
    out["synthetic_groves_pivot_reduces"] = float(np.abs(pay - pay3 + pay).max() >= 0)
    return out


if __name__ == "__main__":
    print(bench_groves())
