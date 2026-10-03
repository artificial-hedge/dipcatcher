"""glft_closed_form — when is the GLFT asymptotic quote trustworthy?

The Avellaneda–Stoikov market maker under exponential utility and
fill intensity λ(δ) = A·e^{−kδ} reduces (GLFT 2013, "Dealing with the
inventory risk") to a 2Q+1-dimensional ODE system in θ_q(t):

    γ·θ̇_q = (γσ²/2)·q² − ν·[e^{−k(θ_q−θ_{q−1})} + e^{−k(θ_q−θ_{q+1})}]

    ν = A·γ·k^{k/γ} / (k+γ)^{1+k/γ},   θ_q(T) = 0,

with the inventory-bounded boundary terms dropped at q = ±Q. Optimal
offsets from mid:

    δ*_a,q = δ∞ + θ_q − θ_{q−1},   δ*_b,q = δ∞ + θ_q − θ_{q+1},
    δ∞ = (1/γ)·ln(1 + γ/k).

The GLFT *closed form* linearizes the exponentials
(e^{−kΔ} ≈ 1 − kΔ), giving a symmetric tridiagonal linear system
θ̇ = b + M·θ that diagonalizes in closed form — the formula every
desk quotes. The catch nobody ships: the expansion is valid only
while k|θ_q−θ_{q±1}| ≪ 1 — and since the leading-order inventory
curvature is θ_q ∼ σ²q²(T−t)/2, the true validity axis is
ξ = k·σ²·q_max·T, NOT γ. This module solves BOTH systems and emits
the approximation error in ticks against measured ξ, so a caller
knows when the cheap formula is lying. A seeded Monte-Carlo check
pins the chain: E[−e^{−γ(x+qs)}] under the ODE quotes must equal
−e^{−γθ_0(0)}.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _nu(gamma: float, k: float, A: float) -> float:
    return float(A * gamma * k ** (k / gamma) / (k + gamma) ** (1.0 + k / gamma))


def _check_params(gamma: float, k: float, A: float, sigma: float, T: float, q_max: int) -> None:
    if gamma <= 0 or k <= 0 or A <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("gamma, k, A, sigma, T must all be positive")
    if q_max < 1:
        raise ValueError("q_max must be >= 1")


def _theta_rhs(
    theta: np.ndarray, q_idx: np.ndarray, sigma: float, gamma: float, k: float, nu: float
) -> np.ndarray:
    n = theta.size
    rhs = (sigma * sigma / 2.0) * q_idx.astype(float) ** 2 / gamma
    for i in range(n):
        # q=−Q cannot sell: the θ_{q−1} term is absent (0, not e^0)
        if i > 0:
            rhs[i] -= (nu / gamma) * np.exp(-k * (theta[i] - theta[i - 1]))
        # q=+Q cannot buy: the θ_{q+1} term is absent
        if i < n - 1:
            rhs[i] -= (nu / gamma) * np.exp(-k * (theta[i] - theta[i + 1]))
    return rhs


def solve_theta_ode(
    gamma: float,
    k: float,
    A: float,
    sigma: float,
    T: float,
    q_max: int,
    n_steps: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward RK4 solve of the nonlinear θ system from θ(T)=0.

    Returns (times ascending [0..T], theta[step, q_index]); theta[0] is t=0.
    """
    _check_params(gamma, k, A, sigma, T, q_max)
    nu = _nu(gamma, k, A)
    n = 2 * q_max + 1
    q_idx = np.arange(-q_max, q_max + 1)
    theta = np.zeros(n)
    times = np.linspace(T, 0.0, n_steps + 1)
    out = np.empty((n_steps + 1, n))
    out[0] = theta
    dt = times[1] - times[0]  # negative
    for s in range(n_steps):
        k1 = _theta_rhs(theta, q_idx, sigma, gamma, k, nu)
        k2 = _theta_rhs(theta + 0.5 * dt * k1, q_idx, sigma, gamma, k, nu)
        k3 = _theta_rhs(theta + 0.5 * dt * k2, q_idx, sigma, gamma, k, nu)
        k4 = _theta_rhs(theta + dt * k3, q_idx, sigma, gamma, k, nu)
        theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[s + 1] = theta
    return times[::-1].copy(), out[::-1].copy()


def theta_closed_form(
    gamma: float,
    k: float,
    A: float,
    sigma: float,
    T: float,
    q_max: int,
    t: float = 0.0,
) -> np.ndarray:
    """GLFT linearized-system closed form at time t.

    Linearizing e^{−k(θ_q−θ_{q±1})} ≈ 1 − k(θ_q−θ_{q±1}) gives
    θ̇ = b + Mθ with M symmetric tridiagonal (diag +2c, off-diag −c,
    c = νk/γ; +c on the one-sided boundary diagonals) and
    b_q = σ²q²/(2γ) − 2ν/γ (−ν/γ at boundaries). Backward from
    θ(T)=0: θ(t) = −M⁻¹(I − e^{−M(T−t)})b, evaluated by
    eigendecomposition of M.
    """
    _check_params(gamma, k, A, sigma, T, q_max)
    if not 0.0 <= t <= T:
        raise ValueError("t must lie in [0, T]")
    nu = _nu(gamma, k, A)
    n = 2 * q_max + 1
    q_idx = np.arange(-q_max, q_max + 1)
    tau = T - t
    c = nu * k / gamma
    # linearization: −(ν/γ)e^{−k(θ_q−θ_{q±1})} ≈ −ν/γ + c(θ_q − θ_{q±1})
    # ⇒ M diag +2c (+c at the one-sided boundaries), off-diag −c
    M = (
        np.diag(2.0 * c * np.ones(n))
        + np.diag(-c * np.ones(n - 1), 1)
        + np.diag(-c * np.ones(n - 1), -1)
    )
    M[0, 0] = c  # q=-Q: no sell-side term
    M[n - 1, n - 1] = c  # q=+Q: no buy-side term
    b = (sigma * sigma / 2.0) * q_idx.astype(float) ** 2 / gamma - 2.0 * nu / gamma
    b[0] = (sigma * sigma / 2.0) * q_max**2 / gamma - nu / gamma
    b[n - 1] = b[0]
    evals, evecs = np.linalg.eigh(M)  # M symmetric ⇒ orthonormal U
    # backward solution: θ(t) = −M⁻¹(I − e^{−Mτ})b; M has a kernel
    # mode (constant θ) so take the λ→0 limit τ where needed.
    coeff = np.where(np.abs(evals) < 1e-10, tau, -np.expm1(-evals * tau) / evals)
    return -evecs @ (coeff * (evecs.T @ b))


def quotes_from_theta(
    theta: np.ndarray, gamma: float, k: float, q_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """(ask offsets, bid offsets) per inventory q = −Q..Q from mid."""
    if gamma <= 0 or k <= 0:
        raise ValueError("gamma, k must be positive")
    if theta.size != 2 * q_max + 1:
        raise ValueError("theta length must be 2*q_max + 1")
    d_inf = np.log1p(gamma / k) / gamma
    ask = np.full(theta.size, d_inf)
    bid = np.full(theta.size, d_inf)
    for i in range(theta.size):
        if i > 0:
            ask[i] += theta[i] - theta[i - 1]
        else:
            ask[i] += np.nan  # cannot sell at q=−Q
        if i < theta.size - 1:
            bid[i] += theta[i] - theta[i + 1]
        else:
            bid[i] += np.nan
    return ask, bid


def mc_value_check(
    gamma: float,
    k: float,
    A: float,
    sigma: float,
    T: float,
    q_max: int,
    n_paths: int = 3000,
    n_steps: int = 200,
    seed: int = 7,
) -> dict[str, float]:
    """Simulate the MM under ODE quotes; compare realized E[−e^{−γ(x+qs)}]
    to u(0,0,0,s)=−e^{−γθ_0(0)}. Returns the two values + |diff|."""
    times, thetas = solve_theta_ode(gamma, k, A, sigma, T, q_max, n_steps=400)
    theta0 = thetas[0]
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    utils = np.empty(n_paths)
    for p in range(n_paths):
        x = 0.0
        q = 0
        s = 0.0
        for i in range(n_steps):
            t = i * dt
            ti = int(np.searchsorted(times, t, side="right") - 1)
            ask, bid = quotes_from_theta(theta0 if i == 0 else thetas[ti], gamma, k, q_max)
            qi = q + q_max
            lam_a = A * np.exp(-k * ask[qi]) if q > -q_max else 0.0
            lam_b = A * np.exp(-k * bid[qi]) if q < q_max else 0.0
            u = rng.uniform()
            if u < lam_a * dt and q > -q_max:
                x += s + ask[qi]
                q -= 1
            elif u < (lam_a + lam_b) * dt and q < q_max:
                x -= s - bid[qi]
                q += 1
            s += sigma * np.sqrt(dt) * rng.standard_normal()
        utils[p] = -np.exp(-gamma * (x + q * s))
    u_pde = -np.exp(-gamma * theta0[q_max])
    return {
        "value_mc": float(utils.mean()),
        "value_pde": float(u_pde),
        "abs_diff": float(abs(utils.mean() - u_pde)),
        "mc_se": float(utils.std(ddof=1) / np.sqrt(n_paths)),
    }


def glft_bench(
    gammas: tuple[float, ...] = (0.05, 0.2),
    A: float = 1.5,
    k: float = 1.5,
    sigma: float = 0.02,
    Ts: tuple[float, ...] = (0.02, 0.2, 1.0),
    q_max: int = 3,
    tick: float = 0.01,
    seed: int = 7,
) -> dict[str, Any]:
    """Approximation error of the GLFT closed form vs the ODE truth.

    The expansion parameter is ξ = k·max_q|θ_q−θ_{q±1}|, driven by the
    ν-coupling integrated over T (the forcing level, not σ, sets the
    level); sweep T to move across the validity boundary. Quote error
    is reported in ticks against measured ξ.
    """
    cells: list[dict[str, Any]] = []
    for gamma in gammas:
        for T in Ts:
            _, thetas = solve_theta_ode(gamma, k, A, sigma, T, q_max, n_steps=3000)
            theta_lin = theta_closed_form(gamma, k, A, sigma, T, q_max, t=0.0)
            ask_o, bid_o = quotes_from_theta(thetas[0], gamma, k, q_max)
            ask_l, bid_l = quotes_from_theta(theta_lin, gamma, k, q_max)
            errs = np.concatenate([(ask_l - ask_o)[1:], (bid_l - bid_o)[:-1]]) / tick
            dtheta = np.abs(np.diff(thetas[0])).max()
            cells.append(
                {
                    "gamma": gamma,
                    "sigma": sigma,
                    "T": T,
                    "xi": float(k * dtheta),
                    "theta_max_abs_ode": float(np.abs(thetas[0]).max()),
                    "theta_max_abs_lin": float(np.abs(theta_lin).max()),
                    "quote_err_ticks_max": float(np.abs(errs).max()),
                    "quote_err_ticks_mean": float(np.abs(errs).mean()),
                }
            )
    small = [c for c in cells if c["xi"] < 0.05]
    ok = bool(small) and all(c["quote_err_ticks_max"] < 1.0 for c in small)
    mc = mc_value_check(gammas[0], k, A, sigma, Ts[-1], q_max, seed=seed)
    payload: dict[str, Any] = {
        "kind": "glft_bench",
        "schema": "glft_bench.v1",
        "cells": cells,
        "mc_check": mc,
        "A": A,
        "k": k,
        "sigma": sigma,
        "Ts": list(Ts),
        "q_max": q_max,
        "tick": tick,
        "claim": "glft_closed_form_validity_mapped",
        "interpretation": (
            "θ_lin is the textbook closed form; quote_err shows the tick "
            "error vs the true nonlinear solve, indexed by the real "
            "expansion parameter ξ=k·max|Δθ| (driven by the forcing "
            "timescale ν·T/γ, not σ); the grid sweeps T across it. "
            "ok=True iff every ξ<0.05 cell is within one tick."
        ),
        "ok": ok,
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
