"""Triple difference (DDD) estimation (SYNTHETIC).

The DDD estimator differences out a control group's post-shift across
two additional margins (e.g. treated-vs-untreated unit and
affected-vs-unaffected subgroup), netting out shocks that hit both
subgroups or both unit types. Estimated as the third interaction
coefficient in a saturated regression, or directly as the eight-cell
contrast.

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure effect recovery on generated
three-way panels — never market evidence.

References:
- Gruber, J. (1994). The incidence of mandated maternity benefits.
  *AER* 84, 622-641 — canonical DDD application.
- Olden, A., Møen, J. (2022). The triple difference estimator.
  *Econometrics Journal* 25, 531-553 — identification assumptions
  and comparison with DiD.
- Yelowitz, A. (1995). The Medicaid notch, labor supply, and welfare
  participation. *QJE* 110 — early triple-difference design.
- Wooldridge, J. M. (2023). Simple approaches to nonlinear
  difference-in-differences with panel data. *Econometrics Journal*
  — saturated-model machinery.

Composition: pure numpy — saturated OLS with cluster-free HC1 SE,
eight-cell contrast cross-check; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]


def _as1(v: FloatArray, name: str) -> FloatArray:
    a = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 1-D vector required")
    return a


def triple_difference(
    y: FloatArray,
    post: FloatArray,
    treat: FloatArray,
    elig: FloatArray,
) -> dict[str, float]:
    """DDD via saturated interaction regression.

    ``y = a + p·post + t·treat + e·elig + pt·post·treat +
    pe·post·elig + te·treat·elig + ddd·post·treat·elig + ε``;
    ``ddd`` is the effect. post/treat/elig are binary indicators."""
    yy = _as1(y, "y")
    p = _as1(post, "post")
    t = _as1(treat, "treat")
    e = _as1(elig, "elig")
    n = yy.size
    if p.size != n or t.size != n or e.size != n or n < 40:
        raise ValueError("equal-length y/post/treat/elig, n>=40")
    for name, v in (("post", p), ("treat", t), ("elig", e)):
        if not np.isin(np.unique(v), [0.0, 1.0]).all():
            raise ValueError(f"{name}: must be binary 0/1")

    xmat = np.column_stack([np.ones(n), p, t, e, p * t, p * e, t * e, p * t * e])
    if np.linalg.matrix_rank(xmat) < xmat.shape[1]:
        raise ValueError("design singular — need all eight cells populated")
    xt = xmat.T @ xmat
    b = np.linalg.solve(xt, xmat.T @ yy)
    resid = yy - xmat @ b
    meat = xmat.T @ ((resid**2)[:, None] * xmat)
    xtxi = np.linalg.inv(xt)
    vcov = xtxi @ meat @ xtxi * n / (n - xmat.shape[1])
    ddd = float(b[-1])
    se = float(math.sqrt(max(vcov[-1, -1], 0.0)))
    z = ddd / max(se, 1e-12)

    # eight-cell contrast cross-check
    cells = {}
    for pp in (0, 1):
        for tt in (0, 1):
            for ee in (0, 1):
                m = (p == pp) & (t == tt) & (e == ee)
                if m.sum() == 0:
                    raise ValueError("empty cell in DDD design")
                cells[(pp, tt, ee)] = float(yy[m].mean())
    contrast = (
        cells[(1, 1, 1)]
        - cells[(0, 1, 1)]
        - cells[(1, 0, 1)]
        + cells[(0, 0, 1)]
        - cells[(1, 1, 0)]
        + cells[(0, 1, 0)]
        + cells[(1, 0, 0)]
        - cells[(0, 0, 0)]
    )

    return {
        "ddd": ddd,
        "se": se,
        "z": float(z),
        "p_value": float(2 * (1 - norm.cdf(abs(z)))),
        "cell_contrast": float(contrast),
        "n": float(n),
        "r2": float(1 - float(resid @ resid) / float(((yy - yy.mean()) ** 2).sum())),
    }


def synth_ddd(
    n_per_cell: int = 300,
    effect: float = 1.0,
    cell_confound: float = 0.8,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """DDD DGP: post shocks hit both eligibility groups and both unit
    types; only the treated+eligible+post cell gains ``effect``.
    ``cell_confound`` adds a post shock to treated units regardless
    of eligibility — differenced out by DDD, biasing plain DiD."""
    rng = np.random.default_rng(seed)
    rows = []
    for pp in (0, 1):
        for tt in (0, 1):
            for ee in (0, 1):
                mu = (
                    0.3 * pp  # common post drift
                    + 0.2 * tt  # treated baseline
                    + 0.4 * ee  # eligible baseline
                    + cell_confound * pp * tt  # confounded treated shock
                    + effect * pp * tt * ee
                )
                rows.append(
                    np.column_stack(
                        [
                            mu + rng.normal(0.0, 0.5, n_per_cell),
                            np.full(n_per_cell, pp),
                            np.full(n_per_cell, tt),
                            np.full(n_per_cell, ee),
                        ]
                    )
                )
    arr = np.vstack(rows)
    return {
        "y": arr[:, 0],
        "post": arr[:, 1],
        "treat": arr[:, 2],
        "elig": arr[:, 3],
        "effect_true": np.array([effect]),
    }


def bench_triple_difference(
    seed: int = 20261231 + 211,
) -> dict[str, float]:
    """DDD self-check: ddd recovers the cell-specific effect while the
    naive DiD on eligibles is confounded by the shared post shock.
    All ``synthetic_*``."""
    d = synth_ddd(effect=1.0, seed=seed)
    out = triple_difference(
        np.asarray(d["y"]),
        np.asarray(d["post"]),
        np.asarray(d["treat"]),
        np.asarray(d["elig"]),
    )
    # naive DiD on eligibles only (confounded)
    m = np.asarray(d["elig"]) == 1.0
    y_, p_, t_ = (
        np.asarray(d["y"])[m],
        np.asarray(d["post"])[m],
        np.asarray(d["treat"])[m],
    )
    xn = np.column_stack([np.ones(m.sum()), p_, t_, p_ * t_])
    bn = np.linalg.lstsq(xn, y_, rcond=None)[0]
    naive_did = float(bn[-1])

    d0 = synth_ddd(effect=0.0, seed=seed + 1)
    out0 = triple_difference(
        np.asarray(d0["y"]),
        np.asarray(d0["post"]),
        np.asarray(d0["treat"]),
        np.asarray(d0["elig"]),
    )
    out_b = triple_difference(
        np.asarray(d["y"]),
        np.asarray(d["post"]),
        np.asarray(d["treat"]),
        np.asarray(d["elig"]),
    )

    ddd = float(out["ddd"])
    return {
        "synthetic_ddd": ddd,
        "synthetic_ddd_err": float(abs(ddd - 1.0)),
        "synthetic_se": float(out["se"]),
        "synthetic_z": float(out["z"]),
        "synthetic_naive_did": naive_did,
        "synthetic_naive_bias": float(abs(naive_did - 1.0)),
        "synthetic_beats_naive": float(abs(ddd - 1.0) < abs(naive_did - 1.0)),
        "synthetic_null_ddd": float(abs(out0["ddd"])),
        "synthetic_detects": float(abs(ddd - 1.0) < 0.2 and float(out["p_value"]) < 0.05),
        "synthetic_determinism": float(ddd == float(out_b["ddd"])),
    }
