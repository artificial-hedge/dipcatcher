"""Sign-restricted structural VAR (Uhlig 2005) (SYNTHETIC).

Estimate a reduced-form VAR, draw orthogonal rotations Q via QR of
random Gaussian matrices, and keep draws whose impulse responses to
the target shock satisfy a set of sign restrictions over a restriction
horizon. Reports the median IRF path and the share of accepted draws —
a pure sign-identification object (no point identification claimed).

All estimators fail closed (ValueError) on degenerate input.

Honesty: synthetic benches measure sign-restriction acceptance and
IRF recovery on generated VAR panels — never market evidence.

References:
- Uhlig (2005). What are the effects of monetary policy on output?
  *J. Monetary Economics* 52, 381-419.
- Arias, Rubio-Ramírez, Waggoner (2018). Inference based on SVARs
  identified with sign and zero restrictions. *Econometrica* 86.
- Lütkepohl (2005). *New Introduction to Multiple Time Series
  Analysis*, ch. 2-3 (VAR MA machinery).
- Rubio-Ramírez, Waggoner, Zha (2010). Structural VARs: theory of
  identification and algorithms. *RES* 77.

Composition: pure numpy — VAR OLS, companion-matrix MA inversion,
QR rotation draws; deterministic ``np.random.default_rng``; no new
dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as2(m: FloatArray, name: str) -> FloatArray:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite 2-D matrix required")
    return a


def var_ols(y: FloatArray, p: int = 1) -> dict[str, FloatArray | int]:
    """Reduced-form VAR(p): y_t = c + A1 y_{t-1} + ... + Ap y_{t-p} + u_t."""
    a = _as2(y, "y")
    t, k = a.shape
    if t < p + 3 or k < 2 or p < 1:
        raise ValueError("need t>=p+3, k>=2, p>=1")
    Y = a[p:]
    X = np.column_stack([np.ones(t - p)] + [a[p - i : t - i] for i in range(1, p + 1)])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    U = Y - X @ B
    sigma_u = (U.T @ U) / (t - p - 1 - k * p)
    return {
        "B": B,  # (1 + kp, k)
        "sigma_u": sigma_u,
        "residuals": U,
        "n_obs": t - p,
        "p": p,
        "k": k,
    }


def var_ma_coefs(fit: dict[str, FloatArray | int], h: int) -> FloatArray:
    """MA coefficient matrices Θ_0..Θ_h via companion-form recursion."""
    B = np.asarray(fit["B"])
    k = int(fit["k"])
    p = int(fit["p"])
    A = [B[1 + i * k : 1 + (i + 1) * k].T for i in range(p)]  # row-loadings → A_i
    theta = [np.eye(k)]
    for j in range(1, h + 1):
        m = np.zeros((k, k))
        for i in range(1, min(p, j) + 1):
            m += theta[j - i] @ A[i - 1]
        theta.append(m)
    return np.stack(theta)  # (h+1, k, k)


def _chol_rotation(sigma_u: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Cholesky factor P of Σ_u plus a random orthogonal Q."""
    k = sigma_u.shape[0]
    p_ = np.linalg.cholesky(sigma_u + 1e-10 * np.eye(k))
    g = rng.normal(0.0, 1.0, (k, k))
    q, _ = np.linalg.qr(g)
    q = q * rng.choice([-1.0, 1.0], size=k)  # free column signs
    return p_ @ q


def sign_restricted_irf(
    y: FloatArray,
    shock_col: int,
    target_cols: tuple[int, ...],
    horizon: int = 8,
    restrict_h: int = 2,
    signs: tuple[int, ...] = (1,),
    p: int = 1,
    n_draws: int = 300,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Uhlig sign-restricted IRF to shock ``shock_col``.

    Each draw uses B0 = P·Q (Cholesky × random orthogonal); the shock
    column is B0[:, shock_col]. A draw is accepted iff IRF_j(h') has
    sign ``signs[j]`` for all restricted target rows j over
    h' < restrict_h. Returns accepted-draw share, median IRF path per
    target, and the 16/84 percentiles."""
    a = _as2(y, "y")
    fit = var_ols(a, p=p)
    k = int(fit["k"])
    if not 0 <= shock_col < k:
        raise ValueError("shock_col out of range")
    if len(signs) != len(target_cols) or not signs:
        raise ValueError("signs/target_cols mismatch")
    for c in target_cols:
        if not 0 <= c < k:
            raise ValueError("target_col out of range")

    theta = var_ma_coefs(fit, horizon)
    sigma_u = np.asarray(fit["sigma_u"])
    rng = np.random.default_rng(seed)

    accepted: list[FloatArray] = []
    n_acc = 0
    for _ in range(n_draws):
        b0 = _chol_rotation(sigma_u, rng)
        sh = b0[:, shock_col]  # impact column
        irf_all = np.stack([theta[h] @ sh for h in range(horizon + 1)])  # (h+1, k)
        ok = True
        for col, sg in zip(target_cols, signs, strict=True):
            seg = irf_all[:restrict_h, col]
            if np.any(sg * seg < -1e-10):
                ok = False
                break
        if ok:
            n_acc += 1
            accepted.append(irf_all)

    acc_share = n_acc / n_draws
    if accepted:
        stack = np.stack(accepted)  # (n_acc, h+1, k)
        med = np.median(stack, axis=0)
        lo = np.percentile(stack, 16, axis=0)
        hi = np.percentile(stack, 84, axis=0)
    else:
        med = np.zeros((horizon + 1, k))
        lo = np.zeros_like(med)
        hi = np.zeros_like(med)

    return {
        "accept_share": float(acc_share),
        "n_accepted": float(n_acc),
        "irf_median": med,
        "irf_lo": lo,
        "irf_hi": hi,
        "target_irf": med[:, list(target_cols)],
        "horizon": float(horizon),
        "k": float(k),
    }


def synth_var(
    n: int = 400,
    k: int = 3,
    persistence: float = 0.6,
    shock_response: float = 1.2,
    seed: int = 0,
) -> dict[str, FloatArray]:
    """VAR(1): first variable is the policy instrument; a unit shock to
    it raises variables 1..k-1 persistently. Used to check that sign
    restrictions recover the true response shape."""
    rng = np.random.default_rng(seed)
    A = np.zeros((k, k))
    for i in range(k):
        A[i, i] = persistence
    # shock to var0 propagates positively to var1, var2
    imp = np.eye(k) * 0.6
    imp[1, 0] = shock_response
    imp[2, 0] = shock_response * 0.7
    imp[0, 0] = 1.0
    y = np.zeros((n, k))
    for t in range(1, n):
        u = rng.normal(0.0, 1.0, k)
        y[t] = A @ y[t - 1] + imp @ u
    return {"y": y, "A": A, "B0": imp}


def bench_sign_restricted_var(seed: int = 20261231 + 201) -> dict[str, float]:
    """Sign-restriction self-check: positive-shock restriction on var1
    accepts draws whose median IRF is positive and recovers persistence;
    wrong-sign restriction accepts few draws. All ``synthetic_*``."""
    d = synth_var(seed=seed)
    y = np.asarray(d["y"])
    pos = sign_restricted_irf(
        y,
        shock_col=0,
        target_cols=(1,),
        horizon=8,
        restrict_h=3,
        signs=(1,),
        n_draws=250,
        seed=seed + 1,
    )
    neg = sign_restricted_irf(
        y,
        shock_col=0,
        target_cols=(1,),
        horizon=8,
        restrict_h=3,
        signs=(-1,),
        n_draws=250,
        seed=seed + 2,
    )
    pos_b = sign_restricted_irf(
        y,
        shock_col=0,
        target_cols=(1,),
        horizon=8,
        restrict_h=3,
        signs=(1,),
        n_draws=250,
        seed=seed + 1,
    )
    med = np.asarray(pos["target_irf"])[:, 0]
    true_persist = float(np.asarray(d["A"])[1, 1])
    # true IRF of var1 to shock0: shock_response * persistence^h
    irf_true = np.asarray([1.2 * true_persist**h for h in range(9)])
    err = float(np.abs(med - irf_true).mean())

    return {
        "synthetic_accept_share_pos": float(pos["accept_share"]),
        "synthetic_accept_share_neg": float(neg["accept_share"]),
        "synthetic_irf_h1": float(med[1]),
        "synthetic_irf_err": err,
        "synthetic_median_positive_h1": float(med[1] > 0.0),
        "synthetic_detects": float(
            float(pos["accept_share"]) > 0.05 and float(med[1]) > 0.0 and err < 1.0
        ),
        "synthetic_determinism": float(float(pos["accept_share"]) == float(pos_b["accept_share"])),
    }
