"""hawkes_real — self-exciting branching ratio on real tape vs sim.

An exponential-kernel Hawkes process for market-order arrivals has
intensity lambda(t) = mu + sum_{t_i<t} eta*beta*exp(-beta(t - t_i)).
eta is the branching ratio: the mean number of child MOs each MO
spawns. eta ~ 0 is Poisson flow; eta -> 1 is critical clustering;
eta > 1 is non-stationary (reported honestly if found).

The measurement matters because it separates the two rival mechanisms
for order-flow persistence: metaorder splitting (split_flow) produces
persistent signs with piecewise-constant intensity, while genuine
self-excitation produces decaying bursts. Fitting eta on the real tape
and on each sim arm tells us which mechanism the sim's stylized facts
actually live on.

MLE is exact for the exp kernel via the intensity recursion
R_i = exp(-beta*dt_i) * (R_{i-1} + eta*beta), O(n) per evaluation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from quant_fund.microstructure.lobster import EXECUTION, parse_messages
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _hawkes_loglik(params: np.ndarray, times: np.ndarray, horizon_s: float) -> float:
    """Negative log-likelihood of exp-kernel Hawkes on `times` in [0, T]."""
    mu, eta, beta = params
    if mu <= 0 or eta < 0 or beta <= 0:
        return 1e30
    n = times.size
    lam = np.empty(n)
    r = 0.0
    prev = 0.0
    for i in range(n):
        r = math.exp(-beta * (times[i] - prev)) * (r + eta * beta)
        lam[i] = mu + r
        prev = times[i]
        if lam[i] <= 0:
            return 1e30
    tail = eta * float(np.sum(1.0 - np.exp(-beta * (horizon_s - times))))
    ll = float(np.sum(np.log(lam))) - mu * horizon_s - tail
    return -ll if math.isfinite(ll) else 1e30


def hawkes_fit(times: np.ndarray, horizon_s: float) -> dict[str, Any]:
    """Exp-kernel Hawkes MLE + Poisson baseline delta-loglik."""
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    times = np.sort(times)
    times = times[(times >= 0.0) & (times <= horizon_s)]
    if times.size < 50 or horizon_s <= 0:
        return {"ok": False, "reason": "insufficient_events", "n": int(times.size)}
    mu0 = times.size / horizon_s
    best: dict[str, Any] | None = None
    # grid over beta (decay), joint-optimize (mu, eta) per beta
    for beta0 in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
        res = minimize(
            _hawkes_loglik,
            x0=np.array([mu0 * 0.8, 0.5, beta0]),
            args=(times, horizon_s),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8},
        )
        if best is None or res.fun < float(best["neg_loglik"]):
            best = {
                "neg_loglik": float(res.fun),
                "mu": float(res.x[0]),
                "eta": float(res.x[1]),
                "beta": float(res.x[2]),
            }
    assert best is not None
    poisson_ll = float(np.sum(np.log(mu0 * np.ones_like(times)))) - mu0 * horizon_s
    return {
        "ok": True,
        "n": int(times.size),
        "horizon_s": round(horizon_s, 3),
        "mu": round(best["mu"], 6),
        "branching_ratio_eta": round(best["eta"], 4),
        "decay_beta": round(best["beta"], 4),
        "hawkes_neg_loglik": round(best["neg_loglik"], 3),
        "poisson_neg_loglik": round(-poisson_ll, 3),
        "loglik_gain_vs_poisson": round(-poisson_ll - best["neg_loglik"], 3),
    }


def mo_times_lobster(message_path: Path) -> np.ndarray:
    """Event-time (seconds) of every visible execution on the tape."""
    times = []
    for ev in parse_messages(message_path):
        if ev.event_type == EXECUTION:
            times.append(ev.time_s)
    return np.asarray(times)


def mo_times_sim(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> tuple[np.ndarray, float]:
    """MO timestamps (sim clock seconds) under a flow arm."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(horizon):
        sim.step()
    ts = np.asarray([tr.t for tr in sim.trades], dtype=float)
    horizon_s = float(sim._t) if math.isfinite(sim._t) else float(ts.max())
    return ts, horizon_s


def hawkes_real_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Branching ratio on the real tape vs each sim flow. Sealed receipt."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    real_times = mo_times_lobster(msg)
    t0 = float(real_times.min()) if real_times.size else 34200.0
    real = hawkes_fit(real_times - t0, 57600.0 - 34200.0)

    arms: dict[str, Any] = {}
    ts, hz = mo_times_sim(seed=seed)
    arms["iid"] = hawkes_fit(ts, hz)
    ts, hz = mo_times_sim(
        flow=MarkovRegimeFlow(
            states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
            stay_probs=(0.995, 0.985),
            seed=seed + 1,
        ),
        seed=seed + 1,
    )
    arms["regime"] = hawkes_fit(ts, hz)
    ts, hz = mo_times_sim(
        flow=SplitFlow(
            p_start=0.10,
            size_tail=1.2,
            k_min=10,
            k_max=600,
            intensity_mult=3.0,
            seed=seed + 2,
        ),
        seed=seed + 2,
    )
    arms["split"] = hawkes_fit(ts, hz)

    payload: dict[str, Any] = {
        "kind": "hawkes_real",
        "schema": "hawkes_real.v1",
        "ticker": ticker,
        "estimator": "exp-kernel Hawkes MLE: lambda=mu+sum(eta*beta*e^{-beta*dt}); eta=branching ratio",
        "real": real,
        "sim_arms": arms,
        "claim": "branching_ratio_measured_real_vs_sim",
        "interpretation": (
            "eta=0 is Poisson; eta->1 critical self-excitation. The "
            "Poisson delta-loglik quantifies whether clustering is "
            "statistically real, not just nonzero. Split flow should "
            "show intermediate eta — parents burst but with constant "
            "intra-parent intensity, unlike genuine recursive "
            "excitation."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
