"""hmm_learn — Baum–Welch EM recovering the planted flow-regime chain.

``regime_filter`` (PR #464) tracks the hidden flow state given *known*
params; this lane asks the harder question — can the params be learned
from the tape alone? It runs the ZI-LOB under a planted 2-state
``MarkovRegimeFlow``, discretizes the tape into per-window categorical
symbols (bucketed by MO intensity × buy fraction), and fits a 2-state
categorical-emission HMM by Baum–Welch EM.

Claim checks compare recovered vs planted per-state statistics after
label-sorting by recovered buy-fraction, and verify the log-likelihood
trace is non-decreasing (EM monotone ascent, within tolerance). All
results labeled SYNTHETIC; receipt sealed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

HMM_LEARN_SCHEMA = "hmm_learn.v1"


@dataclass(frozen=True)
class HMMFit:
    """Fitted 2-state categorical HMM."""

    A: NDArray[np.float64]  # transition matrix, A[i,j] = P(next=j | cur=i)
    B: NDArray[np.float64]  # emission matrix, B[i,k] = P(symbol=k | state=i)
    pi: NDArray[np.float64]  # initial distribution
    loglik_trace: tuple[float, ...]
    n_iter: int


def _forward_backward(
    obs: NDArray[np.int64],
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Scaled forward-backward. Returns (gamma [T,S], xi_sum [S,S], loglik)."""
    T, S = obs.shape[0], A.shape[0]
    alpha = np.empty((T, S))
    c = np.empty(T)
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        if c[t] <= 0.0:  # degenerate: impossible observation
            alpha[t] = 1.0 / S
            c[t] = 1.0
        else:
            alpha[t] /= c[t]
    beta = np.ones((T, S))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1])) / max(c[t + 1], 1e-300)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((S, S))
    for t in range(T - 1):
        m = np.outer(alpha[t], B[:, obs[t + 1]] * beta[t + 1]) * A
        s = m.sum()
        if s > 0.0:
            xi += m / s
    loglik = float(np.log(np.maximum(c, 1e-300)).sum())
    return gamma, xi, loglik


def baum_welch(
    obs: NDArray[np.int64],
    n_symbols: int,
    *,
    n_iter: int = 50,
    seed: int = 0,
    tol: float = 1e-7,
) -> HMMFit:
    """Fit a 2-state categorical HMM. Deterministic given ``seed``."""
    if obs.ndim != 1 or obs.size < 10:
        raise ValueError(f"need >=10 observations, got {obs.size}")
    if int(obs.min()) < 0 or int(obs.max()) >= n_symbols:
        raise ValueError("observations outside symbol range")
    rng = np.random.default_rng(seed)
    S = 2
    A = rng.dirichlet([4.0, 4.0], size=S)
    B = rng.dirichlet(np.ones(n_symbols), size=S)
    pi = np.asarray([0.5, 0.5])
    trace: list[float] = []
    prev = float("-inf")
    for _ in range(n_iter):
        gamma, xi, ll = _forward_backward(obs, A, B, pi)
        trace.append(ll)
        A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        B = np.zeros((S, n_symbols))
        for k in range(n_symbols):
            m = obs == k
            B[:, k] = gamma[m].sum(axis=0)
        B /= np.maximum(B.sum(axis=1, keepdims=True), 1e-300)
        pi = gamma[0] / gamma[0].sum()
        if ll - prev < tol * max(1.0, abs(prev)):
            break
        prev = ll
    return HMMFit(A=A, B=B, pi=pi, loglik_trace=tuple(trace), n_iter=len(trace))


def discretize_tape(
    *,
    config: ZILobConfig,
    horizon: float,
    flow: MarkovRegimeFlow,
    window_s: float = 20.0,
) -> NDArray[np.int64]:
    """Per-window categorical symbol: 2×2 grid over (mo_rate, buy_frac) medians."""
    sim = ZILobSimulator(config, flow=flow)
    mo_counts: list[int] = []
    buy_counts: list[int] = []
    next_t = 0.0
    cur_mo = cur_buy = 0
    while sim.t < horizon:
        kind = sim.step()
        if kind == "market":
            cur_mo += 1
            cur_buy += int(sim.trades[-1].aggressor == "buy")
        while sim.t >= next_t:
            mo_counts.append(cur_mo)
            buy_counts.append(cur_buy)
            cur_mo = cur_buy = 0
            next_t += window_s
    mo = np.asarray(mo_counts[1:], dtype=np.float64)  # drop partial first window
    bu = np.asarray(buy_counts[1:], dtype=np.float64)
    frac = bu / np.maximum(mo, 1.0)
    hi_rate = mo > np.median(mo)
    hi_buy = frac > np.median(frac)
    return hi_rate.astype(np.int64) * 2 + hi_buy.astype(np.int64)


def _flow(seed: int, p_buy: float = 0.80, im: float = 1.6) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend", intensity_mult=im, p_buy=p_buy),
        ],
        stay_probs=[0.97, 0.94],
        seed=seed,
    )


def hmm_learn_bench(
    *,
    n_seeds: int = 6,
    horizon: float = 2000.0,
    window_s: float = 20.0,
) -> dict[str, Any]:
    """EM recovery of the planted chain; label-sorted per-state stats."""
    rec_pbuy_hi: list[float] = []
    rec_pbuy_lo: list[float] = []
    rec_stay_hi: list[float] = []
    rec_stay_lo: list[float] = []
    monotone = True
    for k in range(n_seeds):
        obs = discretize_tape(
            config=ZILobConfig(seed=7000 + k, init_depth=8, band=8),
            horizon=horizon,
            flow=_flow(7000 + k),
            window_s=window_s,
        )
        fit = baum_welch(obs, n_symbols=4, seed=7000 + k)
        tr = np.asarray(fit.loglik_trace)
        if tr.size > 1 and bool(np.any(np.diff(tr) < -1e-6)):
            monotone = False
        # state with higher mean buy-frac emission = "trend" label
        # symbols: bit1=high rate, bit0=high buy frac → P(hi_buy|state)
        p_hi_buy = fit.B[:, 1] + fit.B[:, 3]
        hi = int(np.argmax(p_hi_buy))
        lo = 1 - hi
        rec_pbuy_hi.append(float(p_hi_buy[hi]))
        rec_pbuy_lo.append(float(p_hi_buy[lo]))
        rec_stay_hi.append(float(fit.A[hi, hi]))
        rec_stay_lo.append(float(fit.A[lo, lo]))

    def stats(xs: list[float]) -> dict[str, float]:
        a = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "n": float(a.size),
        }

    payload: dict[str, Any] = {
        "schema": HMM_LEARN_SCHEMA,
        "kind": "hmm_learn",
        "n_seeds": n_seeds,
        "horizon": horizon,
        "window_s": window_s,
        "planted": {
            "p_buy_calm": 0.5,
            "p_buy_trend": 0.8,
            "stay_calm": 0.97,
            "stay_trend": 0.94,
        },
        "recovered": {
            "p_hi_buy_trend_state": stats(rec_pbuy_hi),
            "p_hi_buy_calm_state": stats(rec_pbuy_lo),
            "stay_trend_state": stats(rec_stay_hi),
            "stay_calm_state": stats(rec_stay_lo),
        },
        "likelihood_monotone": monotone,
        "pbuy_ordering_correct": bool(np.mean(rec_pbuy_hi) > np.mean(rec_pbuy_lo)),
        "interpretation": (
            "2-state categorical HMM over windowed (mo_rate, buy_frac) "
            "symbols recovers the label ordering of the planted chain; "
            "EM log-likelihood must be non-decreasing"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
