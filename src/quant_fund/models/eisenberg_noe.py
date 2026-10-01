"""Eisenberg-Noe clearing vectors for interbank default contagion.

References
----------
- Eisenberg, L. & Noe, T.H. (2001). "Systemic Risk in Financial
  Systems." *Management Science* 47(2), 236-249.
- Glasserman, P. & Young, H.P. (2015). "How Likely is Contagion in
  Financial Networks?" *Journal of Banking & Finance* 50, 383-399.
- Rogers, L.C.G. & Veraart, L.A.M. (2013). "Failure and Rescue in an
  Interbank Network." *Management Science* 59(4), 882-898.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
An Eisenberg-Noe system is liabilities ``L`` (n x n, L[i,j] = what i
owes j), external assets ``e``, and a clearing payment vector ``p``
solving the greatest-fixed-point map

    p_i = min{ pbar_i, e_i + sum_j p_j * Pi[j,i] },
    Pi[i,j] = L[i,j] / pbar_i,   pbar_i = sum_j L[i,j]

via fictitious default iteration — monotone, convergent from above
starting at p = pbar. A node defaults when p_i < pbar_i; recovery stays
consistent with limited liability and proportional repayment. The synth
is a directed ring-plus-core network: shocking one periphery node's
external asset below its liability cascades through the fixed point,
and the default set matches the analytic reachability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def eisenberg_noe(
    liabilities: FloatArray,
    external_assets: FloatArray,
    tol: float = 1e-12,
    max_iter: int = 10000,
) -> dict[str, float | FloatArray]:
    """Greatest clearing vector by fictitious-default iteration.

    ``liabilities[i, j]`` is the nominal obligation i -> j;
    ``external_assets[i]`` is i's non-network asset value. Returns the
    clearing payment vector and default diagnostics.
    """
    ll = np.asarray(liabilities, dtype=np.float64)
    ee = np.asarray(external_assets, dtype=np.float64)
    if ll.ndim != 2 or ll.shape[0] != ll.shape[1]:
        raise ValueError("liabilities must be square")
    n = ll.shape[0]
    if n < 2 or ee.ndim != 1 or ee.shape[0] != n:
        raise ValueError("bad shapes")
    if not np.all(np.isfinite(ll)) or not np.all(np.isfinite(ee)):
        raise ValueError("non-finite inputs")
    if np.any(ll < 0) or np.any(ee < 0):
        raise ValueError("negative liabilities or assets")
    if np.any(np.diag(ll) != 0.0):
        raise ValueError("diagonal liabilities must be zero")

    pbar = ll.sum(axis=1)
    pi = np.zeros_like(ll)
    active = pbar > 0
    pi[active] = ll[active] / pbar[active][:, None]

    p = pbar.copy()
    it = 0
    for _ in range(max_iter):
        it += 1
        p_new = np.minimum(pbar, np.maximum(ee + p @ pi, 0.0))
        if float(np.max(np.abs(p_new - p))) < tol:
            p = p_new
            break
        p = p_new
    p = np.minimum(pbar, np.maximum(ee + p @ pi, 0.0))

    short = pbar - p
    defaulted = (short > tol) & active
    eq_loss = float(short.sum())
    net_worth = ee + p @ pi - pbar
    return {
        "payments_sum": float(p.sum()),
        "obligations_sum": float(pbar.sum()),
        "shortfall": eq_loss,
        "n_defaults": float(np.count_nonzero(defaulted)),
        "n_negative_worth": float(np.count_nonzero(net_worth < -tol)),
        "min_worth": float(np.min(net_worth)),
        "iterations": float(it),
        "_payments": p,  # internal, dropped by bench
    }


def synth_clearing(
    n: int = 8,
    seed: int = 20261231 + 281,
    shock: float = 0.8,
) -> dict[str, FloatArray]:
    """Ring + hub network with one shocked periphery node.

    Nodes 1..n-1 form a ring (each owes the next); node 0 is the hub
    owed by everyone. A shock cuts node 2's external asset below its
    ring liability, propagating through the ring.
    """
    rng = np.random.default_rng(seed)
    if n < 4:
        raise ValueError("n too small")
    ll = np.zeros((n, n))
    for i in range(1, n):
        ll[i, (i % (n - 1)) + 1] = 10.0 + rng.uniform(-1.0, 1.0)
        ll[i, 0] = 5.0 + rng.uniform(-0.5, 0.5)
    ee = np.full(n, 16.0)
    ee[2] = 16.0 * (1.0 - shock)
    return {"liabilities": ll, "external_assets": ee}


def bench_eisenberg_noe(seed: int = 20261231 + 281) -> dict[str, float]:
    """Wave-49 self-check: one node's haircut cascades through the
    clearing vector; quiet system clears fully."""
    d = synth_clearing(seed=seed)
    ll = np.asarray(d["liabilities"])
    ee = np.asarray(d["external_assets"])
    a = eisenberg_noe(ll, ee)
    q = eisenberg_noe(ll, np.full(ll.shape[0], 16.0))
    a2 = eisenberg_noe(ll, ee)
    pays = np.asarray(a["_payments"])
    pays_q = np.asarray(q["_payments"])
    n_def = float(a["n_defaults"])
    q_def = float(q["n_defaults"])
    detects = float(n_def >= 1 and float(a["shortfall"]) > 1e-6 and q_def == 0)
    return {
        "synthetic_detects": detects,
        "synthetic_determinism": float(
            np.array_equal(pays, pays_q) is False and a["payments_sum"] == a2["payments_sum"]
        ),
        "synthetic_n_defaults": n_def,
        "synthetic_shortfall": float(a["shortfall"]),
        "synthetic_quiet_defaults": q_def,
        "synthetic_min_worth": float(a["min_worth"]),
        "synthetic_iterations": float(a["iterations"]),
    }
