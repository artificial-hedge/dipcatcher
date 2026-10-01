"""Sparse Identification of Nonlinear Dynamics (SINDy).

SINDy discovers governing ODEs ``dx/dt = Theta(x) @ Xi`` from trajectory
data: build a library ``Theta`` of candidate basis functions evaluated on
the state, then fit sparse coefficient matrix ``Xi`` by sequential
thresholded least squares (STLSQ) — iterate ridge regression restricted to
un-thresholded terms, zeroing coefficients below ``lam``. The result is a
parsimonious, *interpretable* dynamical law rather than a black-box fit.

Functions
---------
- :func:`polynomial_library` — basis Theta with deterministic name list.
- :func:`library_eval` — evaluate the same basis on new rows.
- :func:`stlsq` — sequential thresholded least squares.
- :func:`sindy_fit` — numeric derivative + library + STLSQ → SindyResult.
- :func:`sindy_predict` / :func:`simulate_sindy` — integrate the learned ODE.
- :func:`synth_lorenz` / :func:`synth_van_der_pol` — seeded truth systems.
- :func:`bench_sindy` — SYNTHETIC telemetry blob.

References
----------
- Brunton, Proctor & Kutz (2016). Discovering governing equations from
  data by sparse identification of nonlinear dynamical systems.
  *PNAS* 113(15):3932 — arXiv:1509.03580.
- Brunton, Proctor, Kutz & Bischoff (2016). Sparse identification of
  nonlinear dynamics with control (SINDYc). *IFAC NOLCOS* —
  arXiv:1605.06682.
- Kaiser, Kutz & Brunton (2018). Sparse identification of nonlinear
  dynamics for model predictive control in the low-data limit.
  *Proc. R. Soc. A* 474 — arXiv:1711.05501.

Honesty
-------
All bench numbers are SYNTHETIC recovery checks on seeded ODE systems —
they verify that STLSQ recovers sparse coefficient support and that the
discovered law integrates close to the truth. Nothing here claims a real
dynamical law for markets; chaotic systems diverge regardless.

Composition notes
-----------------
- Independent lane: no reuse of ``filters``/``state_space`` (those assume
  a known model class; here the model class itself is identified).
- Reuses repo conventions: deterministic seeds, fail-closed ValueErrors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]


def _as_trajectory(x: FloatArray, name: str = "x", min_rows: int = 8) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < min_rows or a.shape[1] < 1:
        raise ValueError(f"{name}: expected (n>={min_rows}, d>=1) matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    return a


def _monomial_powers(d: int, order: int) -> list[tuple[int, ...]]:
    """All exponent tuples (length d) of total degree 0..order, lexicographic."""
    powers: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int, k: int) -> None:
        if k == d - 1:
            for e in range(remaining + 1):
                powers.append(tuple(prefix + [e]))
            return
        for e in range(remaining + 1):
            rec(prefix + [e], remaining - e, k + 1)

    rec([], order, 0)
    return powers


def _pow_names(powers: list[tuple[int, ...]]) -> list[str]:
    names = []
    for p in powers:
        if all(e == 0 for e in p):
            names.append("1")
            continue
        parts = [f"x{j}^{e}" if e > 1 else f"x{j}" for j, e in enumerate(p) if e > 0]
        names.append("*".join(parts))
    return names


def polynomial_library(
    x: FloatArray, order: int = 3
) -> tuple[FloatArray, list[str], list[tuple[int, ...]]]:
    """Evaluate the polynomial basis on rows of ``x`` (n, d) → (Theta, names, powers)."""
    x = _as_trajectory(x)
    if order < 1 or order > 6:
        raise ValueError("order must be in [1, 6]")
    n, d = x.shape
    powers = _monomial_powers(d, order)
    theta = np.empty((n, len(powers)))
    for j, p in enumerate(powers):
        col = np.ones(n)
        for k, e in enumerate(p):
            if e:
                col = col * x[:, k] ** e
        theta[:, j] = col
    return theta, _pow_names(powers), powers


def library_eval(x: FloatArray, powers: list[tuple[int, ...]]) -> FloatArray:
    """Re-evaluate an existing basis (given by ``powers``) on new rows."""
    x = _as_trajectory(x, min_rows=1)
    n, d = x.shape
    if any(len(p) != d for p in powers):
        raise ValueError("powers dimension does not match x")
    theta = np.empty((n, len(powers)))
    for j, p in enumerate(powers):
        col = np.ones(n)
        for k, e in enumerate(p):
            if e:
                col = col * x[:, k] ** e
        theta[:, j] = col
    return theta


def stlsq(
    theta: FloatArray,
    dxt: FloatArray,
    lam: float = 0.1,
    ridge: float = 1e-6,
    max_iter: int = 20,
) -> tuple[FloatArray, list[float], int]:
    """Sequential thresholded least squares.

    Iterates: ridge-solve each output column restricted to its active
    support, then hard-threshold ``|xi| < lam`` out of the support.
    Returns ``(Xi, mse_path, n_iters)`` with ``Xi`` shape (p, m).
    """
    theta = _as_trajectory(theta, "theta", min_rows=4)
    dxt = _as_trajectory(dxt, "dxt", min_rows=4)
    if theta.shape[0] != dxt.shape[0]:
        raise ValueError("theta and dxt row counts differ")
    if lam < 0:
        raise ValueError("lam must be non-negative")
    p = theta.shape[1]
    m = dxt.shape[1]
    active = np.ones((p, m), dtype=bool)
    xi = np.zeros((p, m))
    mse_path: list[float] = []
    for _ in range(max_iter):
        new_xi = np.zeros((p, m))
        for col in range(m):
            idx = active[:, col]
            if not idx.any():
                continue
            tt = theta[:, idx]
            gram = tt.T @ tt + ridge * np.eye(idx.sum())
            new_xi[np.ix_(idx, [col])] = np.linalg.solve(gram, tt.T @ dxt[:, [col]])
        small = np.abs(new_xi) < lam
        new_active = active & ~small
        resid = dxt - theta @ new_xi
        mse = float(np.mean(resid * resid))
        mse_path.append(mse)
        xi = new_xi
        if np.array_equal(new_active, active):
            active = new_active
            break
        active = new_active
    return xi, mse_path, len(mse_path)


@dataclass(frozen=True)
class SindyResult:
    """Fitted SINDy model.

    Attributes
    ----------
    coefficients:
        ``Xi`` with shape (n_library, n_states) — column k is the sparse
        law for ``dx_k/dt``.
    powers / names:
        Library monomials and human-readable names.
    rmse:
        Training RMSE across all state channels.
    n_nonzero:
        Number of nonzero coefficients.
    """

    coefficients: FloatArray
    powers: list[tuple[int, ...]]
    names: list[str]
    rmse: float
    n_nonzero: int


def _rhs_from(
    xi: FloatArray, powers: list[tuple[int, ...]]
) -> Callable[[float, NDArray[np.float64]], list[float]]:
    def rhs(_t: float, y: NDArray[np.float64]) -> list[float]:
        row = np.asarray(y, dtype=np.float64).reshape(1, -1)
        th = library_eval(row, powers)
        return (th @ xi).ravel().tolist()

    return rhs


def sindy_fit(
    x: FloatArray,
    dt: float,
    order: int = 3,
    lam: float = 0.1,
    ridge: float = 1e-6,
) -> SindyResult:
    """Fit SINDy: centered finite-difference derivative + STLSQ."""
    x = _as_trajectory(x)
    if dt <= 0:
        raise ValueError("dt must be positive")
    dxt = np.gradient(x, dt, axis=0)
    theta, names, powers = polynomial_library(x, order=order)
    xi, mse_path, _it = stlsq(theta, dxt, lam=lam, ridge=ridge)
    rmse = float(math.sqrt(mse_path[-1])) if mse_path else float("nan")
    return SindyResult(
        coefficients=xi,
        powers=powers,
        names=names,
        rmse=rmse,
        n_nonzero=int(np.count_nonzero(xi)),
    )


def sindy_predict(
    result: SindyResult, x0: FloatArray, t: FloatArray, rtol: float = 1e-7
) -> FloatArray:
    """Integrate the discovered ODE from ``x0`` at times ``t``."""
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must have at least 2 points")
    sol = solve_ivp(
        _rhs_from(result.coefficients, result.powers),
        (float(t[0]), float(t[-1])),
        x0.tolist(),
        t_eval=t,
        rtol=rtol,
        atol=1e-9,
    )
    return np.asarray(sol.y, dtype=np.float64).T


def simulate_sindy(result: SindyResult, x0: FloatArray, n: int, dt: float) -> FloatArray:
    """Integrate n steps of size dt."""
    if n < 2:
        raise ValueError("n must be >= 2")
    t = np.arange(n, dtype=np.float64) * dt
    return sindy_predict(result, x0, t)


def _lorenz63(
    sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> Callable[[float, NDArray[np.float64]], list[float]]:
    def rhs(_t: float, y: NDArray[np.float64]) -> list[float]:
        x_, y_, z_ = y
        return [sigma * (y_ - x_), x_ * (rho - z_) - y_, x_ * y_ - beta * z_]

    return rhs


def _van_der_pol(
    mu: float = 1.5,
) -> Callable[[float, NDArray[np.float64]], list[float]]:
    def rhs(_t: float, y: NDArray[np.float64]) -> list[float]:
        x_, v_ = y
        return [v_, mu * (1.0 - x_ * x_) * v_ - x_]

    return rhs


def synth_lorenz(
    n: int = 2000,
    dt: float = 0.005,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int = 0,
    t_transient: float = 10.0,
) -> tuple[FloatArray, FloatArray]:
    """True Lorenz-63 trajectory after a transient burn-in."""
    del seed  # deterministic system; kept for signature symmetry
    sol = solve_ivp(
        _lorenz63(sigma, rho, beta),
        (0.0, t_transient + n * dt),
        [1.0, 1.0, 1.0],
        t_eval=np.linspace(t_transient, t_transient + n * dt, n),
        rtol=1e-9,
        atol=1e-11,
    )
    x = np.asarray(sol.y, dtype=np.float64).T
    t = np.asarray(sol.t, dtype=np.float64)
    return x, t


def synth_van_der_pol(
    n: int = 1500, dt: float = 0.01, mu: float = 1.5, seed: int = 0
) -> tuple[FloatArray, FloatArray]:
    """True Van der Pol trajectory."""
    del seed
    sol = solve_ivp(
        _van_der_pol(mu),
        (0.0, n * dt),
        [2.0, 0.0],
        t_eval=np.arange(n, dtype=np.float64) * dt,
        rtol=1e-9,
        atol=1e-11,
    )
    return np.asarray(sol.y, dtype=np.float64).T, np.asarray(sol.t, dtype=np.float64)


def _true_lorenz_powers() -> tuple[dict[tuple[int, ...], float], ...]:
    """True sparse coefficient maps for Lorenz-63."""
    s, r, b = 10.0, 28.0, 8.0 / 3.0
    # dx = s(y - x); dy = x(r - z) - y; dz = xy - b z
    return (
        {(1, 0, 0): -s, (0, 1, 0): s},
        {(1, 0, 0): r, (0, 1, 0): -1.0, (1, 0, 1): -1.0},
        {(1, 1, 0): 1.0, (0, 0, 1): -b},
    )


def _coef_map(
    xi: FloatArray, powers: list[tuple[int, ...]]
) -> tuple[dict[tuple[int, ...], float], ...]:
    return tuple(
        {powers[i]: xi[i, k] for i in range(len(powers)) if abs(xi[i, k]) > 1e-8}
        for k in range(xi.shape[1])
    )


def bench_sindy(seed: int = 20261231 + 151) -> dict[str, float]:
    """SYNTHETIC SINDy recovery telemetry."""
    rng = np.random.default_rng(seed)
    x_l, _t = synth_lorenz(n=3000, dt=0.004)
    res = sindy_fit(x_l, dt=0.004, order=3, lam=0.05)
    maps = _coef_map(res.coefficients, res.powers)
    truth = _true_lorenz_powers()
    # coefficient relerr over the union of true keys
    errs = []
    for k in range(3):
        keys = set(truth[k]) | set(maps[k])
        for p in keys:
            errs.append(abs(maps[k].get(p, 0.0) - truth[k].get(p, 0.0)) / abs(truth[k].get(p, 1.0)))
    coef_relerr = float(np.mean(errs))
    n_true = sum(len(tk) for tk in truth)
    n_nonzero_err = abs(res.n_nonzero - n_true) / max(n_true, 1)
    # trajectory error: integrate discovered law 200 steps
    pred = simulate_sindy(res, x_l[0], n=200, dt=0.004)
    denom = np.maximum(np.abs(x_l[:200]).mean(axis=0), 1e-6)
    traj_relerr = float(np.mean(np.abs(pred - x_l[:200]) / denom))
    # Van der Pol
    x_v, _tv = synth_van_der_pol(n=2000, dt=0.01, mu=1.5)
    res_v = sindy_fit(x_v, dt=0.01, order=3, lam=0.05)
    maps_v = _coef_map(res_v.coefficients, res_v.powers)
    truth_v: tuple[dict[tuple[int, ...], float], ...] = (
        {(0, 1): 1.0},
        {(1, 0): -1.0, (0, 1): 1.5, (2, 1): -1.5},
    )
    errs_v = []
    for k in range(2):
        keys = set(truth_v[k]) | set(maps_v[k])
        for p in keys:
            errs_v.append(
                abs(maps_v[k].get(p, 0.0) - truth_v[k].get(p, 0.0)) / abs(truth_v[k].get(p, 1.0))
            )
    vdp_relerr = float(np.mean(errs_v))
    # determinism: refit identical
    res2 = sindy_fit(x_l, dt=0.004, order=3, lam=0.05)
    deterministic = float(np.array_equal(res.coefficients, res2.coefficients))
    # noise robustness: small noise keeps support overlap
    noise = rng.standard_normal(x_l.shape) * 0.05
    res_n = sindy_fit(x_l + noise, dt=0.004, order=3, lam=0.05)
    maps_n = _coef_map(res_n.coefficients, res_n.powers)
    sup_true = set().union(*[set(tk) for tk in truth])
    sup_found = set().union(*[set(mk) for mk in maps_n])
    jaccard = len(sup_true & sup_found) / max(len(sup_true | sup_found), 1)
    return {
        "synthetic_coef_relerr": coef_relerr,
        "synthetic_vdp_coef_relerr": vdp_relerr,
        "synthetic_n_nonzero_err": n_nonzero_err,
        "synthetic_traj_relerr": traj_relerr,
        "synthetic_sparsity_ratio": res.n_nonzero / res.coefficients.size,
        "synthetic_noisy_support_jaccard": jaccard,
        "synthetic_rmse": res.rmse,
        "synthetic_determinism": deterministic,
    }
