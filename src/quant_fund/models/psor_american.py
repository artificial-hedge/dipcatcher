"""American put via projected SOR (PSOR) on the linear complementarity problem.

Crank-Nicolson spatial discretization of the Black-Scholes operator; each
backward time step solves min∂tV ≥ 0, V ≥ payoff by SOR with projection onto
the obstacle. SYNTHETIC bench: error vs the 2000-step CRR reference, plus the
European put as the honest-negative floor.
"""

import numpy as np

from quant_fund.models._amopt_synth import (
    S0,
    SIG,
    K,
    Q,
    R,
    T,
    crr_price,
    european_put,
)


def _psor_put(ns: int = 200, nt: int = 200, w: float = 1.2, tol: float = 1e-7) -> float:
    smax = 4.0 * K
    dt = T / nt
    s = np.linspace(0.0, smax, ns + 1)
    i = np.arange(ns + 1)
    dt4, dt2 = dt / 4.0, dt / 2.0
    a = dt4 * ((R - Q) * i - SIG**2 * i**2)
    b = 1.0 + dt2 * (SIG**2 * i**2 + R)
    c = -dt4 * ((R - Q) * i + SIG**2 * i**2)
    ap = -a
    bp = 1.0 - dt2 * (SIG**2 * i**2 + R)
    cp = -c
    payoff = np.maximum(K - s, 0.0)
    v = payoff.copy()
    rhs = np.zeros(ns + 1)
    for _n in range(nt, 0, -1):
        rhs[:] = ap * np.roll(v, 1) + bp * v + cp * np.roll(v, -1)
        rhs[0] = 0.0
        rhs[ns] = 0.0
        y = v.copy()
        y[0] = K
        y[ns] = 0.0
        for _it in range(200):
            err = 0.0
            for j in range(1, ns):
                val = (rhs[j] - a[j] * y[j - 1] - c[j] * y[j + 1]) / b[j]
                new = max(y[j] + w * (val - y[j]), payoff[j])
                err = max(err, abs(new - y[j]))
                y[j] = new
            if err < tol:
                break
        v = y
    return float(np.interp(S0, s, v))


def bench_psor_american(seed: int = 4101) -> dict[str, float]:
    del seed
    ref, _bnd = crr_price(2000)
    price = _psor_put()
    eur = european_put()
    return {
        "synthetic_psor_price": price,
        "synthetic_psor_err": abs(price - ref),
        "synthetic_psor_ref": ref,
        "synthetic_psor_eur_floor_err": abs(eur - ref),
    }
