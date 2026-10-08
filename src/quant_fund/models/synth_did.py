"""Synthetic difference-in-differences (SDID) (SYNTHETIC).

Arkhangelsky et al.'s estimator interpolates between synthetic control
and DiD: unit weights reproduce the treated unit's pre-period path from
the donor pool, *and* time weights balance pre-periods against post-
periods; the estimator is a doubly-weighted DID regression on the panel.
Compared with plain SC it is robust to persistent unit-level latent
factors (weights only need to match in a factor sense), and compared
with plain DiD it removes unit heterogeneity the parallel-trends
assumption cannot absorb.

Implementation: constrained least squares on a simplex for the unit
weights (active-set projected gradient on the quadratic pre-period fit),
similar weights for pre periods, then a weighted 2x2 DID contrast of
cell means; inference by placebo (in-time / leave-one-out style unit
permutation) — reported honestly as a randomization p.

Honesty: synthetic benches are correctness probes on generated factor
panels — never market evidence; placebo p-values are Monte-Carlo
approximations with the reported resolution (1/(B+1)).

References:
- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021). Synthetic
  difference-in-differences. *American Economic Review* 111(12).
- Athey, Bayati, Doudchenko, Imbens, Khosravi (2021). Matrix completion
  methods for causal panel data models. *JASA* 116.
- Abadie, Diamond, Hainmueller (2010). Synthetic control methods for
  comparative case studies. *JASA* 105.

Composition: pure numpy — projected-gradient simplex QP for weights;
deterministic ``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_matrix(y: FloatArray, name: str) -> FloatArray:
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite (N, T) matrix required")
    if a.shape[0] < 4 or a.shape[1] < 6:
        raise ValueError(f"{name}: need >= 4 units and >= 6 periods")
    return a


def _simplex_qp(
    target: FloatArray, pool: FloatArray, lam: float = 1e-4, iters: int = 400
) -> FloatArray:
    """min_w ||pool w - target||² + lam||w||² s.t. w>=0, 1ᵀw=1.

    Accelerated projected gradient with simplex projection (sort-based).
    """
    n = pool.shape[1]
    a = pool.T @ pool + lam * np.eye(n)
    b = pool.T @ target
    w = np.full(n, 1.0 / n)
    step = 1.0 / (float(np.linalg.eigvalsh(a)[-1]) + 1e-12)

    def _proj(v: FloatArray) -> FloatArray:
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho = int(np.flatnonzero(u * np.arange(1, n + 1) > cssv)[-1])
        theta = cssv[rho] / (rho + 1)
        return np.asarray(np.maximum(v - theta, 0.0), dtype=np.float64)

    for _ in range(iters):
        w = _proj(w - step * (a @ w - b))
    return w


def synth_did(
    y: FloatArray,
    treated: FloatArray,
    t0: int,
    *,
    n_placebo: int = 200,
    seed: int = 0,
    lam: float = 1e-3,
) -> dict[str, float | FloatArray]:
    """SDID estimate of the ATT of ``treated`` units from period ``t0``.

    ``y`` is (N, T); ``treated`` is a boolean/0-1 vector length N with at
    least one treated and three donors; ``t0`` splits pre/post periods.
    """
    y = _as_matrix(y, "y")
    n_units, t = y.shape
    tr = np.asarray(treated).astype(bool).ravel()
    if tr.size != n_units or tr.sum() < 1 or tr.sum() > n_units - 3:
        raise ValueError("need >=1 treated and >=3 control units")
    if not (2 <= t0 <= t - 2):
        raise ValueError(f"t0 must be in [2, T-2], got {t0}")

    y_c_pre = y[~tr, :t0]
    y_t_pre = y[tr, :t0]
    # unit weights: match treated-mean pre path with donors
    target_u = y_t_pre.mean(axis=0)
    w_u = _simplex_qp(target_u, y_c_pre.T, lam=lam)
    # time weights: match post-mean with pre periods on the donor pool
    target_t = y[~tr, t0:].mean(axis=1)
    w_t = _simplex_qp(target_t, y_c_pre, lam=lam)

    # doubly weighted DID: contrast of weighted cell means
    def _est(wu: FloatArray, wt: FloatArray) -> float:
        c_pre = wu @ y[~tr, :t0]
        c_post = wu @ y[~tr, t0:]
        t_pre = y[tr, :t0].mean(axis=0)
        t_post = y[tr, t0:].mean(axis=0)
        pre_c, post_c = float(wt @ c_pre), float(c_post.mean())
        pre_t, post_t = float(wt @ t_pre), float(t_post.mean())
        return (post_t - pre_t) - (post_c - pre_c)

    tau = _est(w_u, w_t)

    # placebo inference: re-run with each donor treated as treated
    rng = np.random.default_rng(seed)
    ctrl_idx = np.flatnonzero(~tr)
    n_pl = min(n_placebo, ctrl_idx.size)
    placebos = np.empty(n_pl)
    for j in range(n_pl):
        fake = ctrl_idx[rng.integers(0, ctrl_idx.size)]
        fake_tr = np.zeros(n_units, dtype=bool)
        fake_tr[fake] = True
        keep = ~(tr | fake_tr)
        if keep.sum() < 3:
            placebos[j] = math.nan
            continue
        y_c = y[keep]
        tgt = y[fake_tr, :t0].mean(axis=0)
        wu_j = _simplex_qp(tgt, y_c[:, :t0].T, lam=lam)
        c_pre = wu_j @ y_c[:, :t0]
        c_post = wu_j @ y_c[:, t0:]
        t_pre = y[fake_tr, :t0].mean(axis=0)
        t_post = y[fake_tr, t0:].mean(axis=0)
        placebos[j] = (t_post.mean() - float(w_t @ t_pre)) - (c_post.mean() - float(w_t @ c_pre))
    placebos = placebos[np.isfinite(placebos)]
    if placebos.size < 10:
        raise ValueError("not enough placebo donors")
    se = float(np.std(placebos, ddof=1))
    p = float((np.sum(np.abs(placebos) >= abs(tau)) + 1.0) / (placebos.size + 1.0))

    return {
        "tau": float(tau),
        "se_placebo": se,
        "z_placebo": float(tau / max(se, 1e-12)),
        "p_placebo": p,
        "n_treated": float(tr.sum()),
        "n_placebo": float(placebos.size),
        "unit_weight_max": float(w_u.max()),
        "pre_fit_rmse": float(np.sqrt(np.mean((w_u @ y[~tr, :t0] - target_u) ** 2))),
        "unit_weights": w_u,
        "time_weights": w_t,
    }


def synth_panel(
    n_units: int = 40,
    t: int = 60,
    n_treated: int = 5,
    t0: int | None = None,
    tau: float = 1.0,
    n_factors: int = 2,
    seed: int = 0,
) -> dict[str, FloatArray | np.float64]:
    """Interactive fixed-effects panel with a staggered-constant effect."""
    if t0 is None:
        t0 = int(0.7 * t)
    rng = np.random.default_rng(seed)
    lam = rng.normal(0.0, 1.0, (n_units, n_factors))
    f = rng.normal(0.0, 1.0, (t, n_factors))
    # factor 0 drifts upward after t0 — selection on lam[:,0] breaks
    # plain DiD's parallel trends; SDID unit weights absorb it
    f[:, 0] += np.where(np.arange(t) >= t0, (np.arange(t) - t0) * 0.4, 0.0)
    y = lam @ f.T + rng.normal(0.0, 0.3, (n_units, t))
    treated = np.zeros(n_units, dtype=bool)
    score = lam[:, 0] + rng.normal(0.0, 0.3, n_units)
    treated[np.argsort(score)[-n_treated:]] = True
    y[treated, t0:] += tau
    return {
        "y": y,
        "treated": treated.astype(np.float64),
        "t0": np.float64(t0),
        "tau": np.float64(tau),
        "lam": lam,
        "f": f,
    }


def bench_synth_did(seed: int = 20261231 + 187) -> dict[str, float]:
    """SDID self-check vs plain DiD on a factor panel — the case where
    parallel trends fails and weighting fixes it. All ``synthetic_*``."""
    d = synth_panel(seed=seed, tau=1.5)
    y = np.asarray(d["y"])
    tr = np.asarray(d["treated"]).astype(bool)
    t0 = int(d["t0"])
    tau_true = float(d["tau"])

    est = synth_did(y, tr, t0, n_placebo=120, seed=seed + 1)
    est2 = synth_did(y, tr, t0, n_placebo=120, seed=seed + 1)

    # plain DiD contrast for reference
    did = float((y[tr, t0:].mean() - y[tr, :t0].mean()) - (y[~tr, t0:].mean() - y[~tr, :t0].mean()))

    d0 = synth_panel(seed=seed + 2, tau=0.0)
    est0 = synth_did(
        np.asarray(d0["y"]),
        np.asarray(d0["treated"]).astype(bool),
        int(d0["t0"]),
        n_placebo=120,
        seed=seed + 3,
    )

    return {
        "synthetic_tau_hat": float(est["tau"]),
        "synthetic_tau_true": tau_true,
        "synthetic_tau_err": abs(float(est["tau"]) - tau_true),
        "synthetic_did_err": abs(did - tau_true),
        "synthetic_sdid_beats_did": float(abs(float(est["tau"]) - tau_true) < abs(did - tau_true)),
        "synthetic_p_placebo": float(est["p_placebo"]),
        "synthetic_null_p": float(est0["p_placebo"]),
        "synthetic_pre_fit_rmse": float(est["pre_fit_rmse"]),
        "synthetic_detects": float(est["p_placebo"] < 0.10 and est["tau"] > 0.5 * tau_true),
        "synthetic_determinism": float(
            est["tau"] == est2["tau"] and est["p_placebo"] == est2["p_placebo"]
        ),
    }
