"""vol_signature — realized-volatility signature plot.

Sample the mid series at increasing clock intervals τ and compute the
annualized-style realized variance RV(τ) = Σ r² per unit time. A
flat signature means diffusive prices; an upward slope at fine τ is
the Epps/bid-ask-bounce microstructure-noise signature — the classic
reason realized vol "grows" as you sample faster.

Real tape: mid from consecutive orderbook rows subsampled to τ grids.
Sim: mid sampled per step (dt from the sim's own clock).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import parse_messages, parse_orderbook_row
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

TAUS_S = (0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)


def _rv_at_tau(times: np.ndarray, mids: np.ndarray, tau: float) -> float | None:
    """Realized variance per second sampled on a τ grid (next-sample rule)."""
    if times.size < 2:
        return None
    # bucket boundaries: grid points τ apart across [t0, t_end]
    t0, t_end = float(times[0]), float(times[-1])
    if t_end - t0 < 2 * tau:
        return None
    edges = np.arange(t0, t_end + 1e-9, tau)
    idx = np.searchsorted(times, edges, side="right") - 1
    idx = np.clip(idx, 0, times.size - 1)
    sampled = mids[idx]
    rets = np.diff(sampled)
    rets = rets[np.isfinite(rets)]
    if rets.size < 5:
        return None
    return round(float(rets.var(ddof=0)) / tau, 8)


def _signature(times: list[float], mids: list[float]) -> dict[str, Any]:
    t = np.asarray(times, dtype=float)
    m = np.asarray(mids, dtype=float)
    rvs = {tau: _rv_at_tau(t, m, tau) for tau in TAUS_S}
    have = {str(k): v for k, v in rvs.items() if v is not None}
    finest = min((k for k, v in rvs.items() if v is not None), default=None)
    coarsest = max((k for k, v in rvs.items() if v is not None), default=None)
    ratio = None
    lo = rvs[finest] if finest is not None else None
    hi = rvs[coarsest] if coarsest is not None else None
    if lo is not None and hi is not None:
        ratio = round(lo / hi, 4)
    return {
        "n_samples": int(t.size),
        "span_s": round(float(t[-1] - t[0]), 2) if t.size else 0.0,
        "rv_per_tau": have,
        "rv_fine_to_coarse_ratio": ratio,
    }


def lobster_signature(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    times: list[float] = []
    mids: list[float] = []
    with ob.open() as fo:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(fo), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                mids.append((asks[0][0] + bids[0][0]) / 200.0)
                times.append(ev.time_s)
    return _signature(times, mids)


def sim_signature(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 60000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    times: list[float] = []
    mids: list[float] = []
    for _ in range(horizon):
        sim.step()
        if sim.mid is not None:
            mids.append(sim.mid / cfg.tick)
            times.append(sim.t)
    return _signature(times, mids)


def signature_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Signature-plot comparison real vs sim. Sealed."""
    real = lobster_signature(tape_dir, ticker)
    arms = {
        "iid": sim_signature(seed=seed),
        "regime": sim_signature(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_signature(
            flow=SplitFlow(
                p_start=0.10,
                size_tail=1.2,
                k_min=10,
                k_max=600,
                intensity_mult=3.0,
                seed=seed + 2,
            ),
            seed=seed + 2,
        ),
    }
    divergences = []
    r_ratio = real.get("rv_fine_to_coarse_ratio")
    for name, arm in arms.items():
        a_ratio = arm.get("rv_fine_to_coarse_ratio")
        if r_ratio and a_ratio and abs(a_ratio - r_ratio) / r_ratio > 0.8:
            divergences.append(f"{name}_signature_ratio_{a_ratio:.2f}_vs_{r_ratio}")
    payload: dict[str, Any] = {
        "kind": "vol_signature",
        "schema": "vol_signature.v1",
        "ticker": ticker,
        "taus_s": list(TAUS_S),
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "vol_signature_measured",
        "interpretation": (
            "rv_fine_to_coarse_ratio > 1 = microstructure noise inflating "
            "fine-scale variance (bid-ask bounce / Epps signature); ~1 = "
            "diffusive."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
