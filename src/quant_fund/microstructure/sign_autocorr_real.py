"""sign_autocorr_real — the Lillo–Farmer sign decay on real tape vs sim.

Lillo & Farmer (2004): market-order signs are strongly autocorrelated,
decaying as a power law rho(k) ~ k^-gamma with gamma ~ 0.5 — the
fingerprint of split metaorders. We measure the full autocorrelation
curve on the real LOBSTER tape and on each sim arm, fit log-log slopes
over a mid-lag window, and compare exponents.

This closes the calibration triptych: order_lifetime (queue mechanics),
propagator_real (impact response), sign_autocorr_real (flow memory).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, EXECUTION_HIDDEN, parse_messages
from quant_fund.microstructure.split_flow import SplitFlow, sign_autocorr_curve
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _fit_exponent(curve: dict[str, float]) -> dict[str, Any]:
    """Log-log slope over mid lags where rho > 0 (power-law signature)."""
    pts = [
        (float(lag), v)
        for lag in LAGS
        if not math.isnan(v := curve.get(f"lag{lag}", float("nan"))) and v > 0
    ]
    if len(pts) < 4:
        return {"exponent": None, "n_positive_lags": len(pts)}
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    # fit over lags 4..512 (skip lag-1/2 burst + the noisy tail)
    sel = (lx >= math.log(4)) & (lx <= math.log(512))
    if sel.sum() < 3:
        sel = np.ones_like(lx, dtype=bool)
    slope, intercept = np.polyfit(lx[sel], ly[sel], 1)
    return {
        "exponent": round(-float(slope), 4),
        "fit_intercept": round(float(intercept), 4),
        "n_positive_lags": len(pts),
    }


def lobster_signs(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Exec-sign autocorrelation curve + exponent on the real tape."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    signs = np.asarray(
        [
            -ev.direction
            for ev in parse_messages(msg)
            if ev.event_type in (EXECUTION, EXECUTION_HIDDEN)
        ],
        dtype=float,
    )
    curve = sign_autocorr_curve(signs, LAGS)
    out: dict[str, Any] = {
        "n_execs": int(signs.size),
        "curve": {k: round(v, 5) for k, v in curve.items() if not math.isnan(v)},
    }
    out.update(_fit_exponent(curve))
    return out


def sim_signs(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(horizon):
        sim.step()
    signs = np.asarray([1.0 if tr.aggressor == "buy" else -1.0 for tr in sim.trades])
    curve = sign_autocorr_curve(signs, LAGS)
    out: dict[str, Any] = {
        "n_execs": int(signs.size),
        "curve": {k: round(v, 5) for k, v in curve.items() if not math.isnan(v)},
    }
    out.update(_fit_exponent(curve))
    return out


def sign_autocorr_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Full sign-memory comparison on the real tape vs arms. Sealed."""
    real = lobster_signs(tape_dir, ticker)
    arms = {
        "iid": sim_signs(seed=seed),
        "regime": sim_signs(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_signs(
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
    for name, arm in arms.items():
        if arm["exponent"] is not None and real["exponent"] is not None:
            gap = abs(arm["exponent"] - real["exponent"])
            if gap > 0.2:
                divergences.append(f"{name}_exponent_gap_{gap:.2f}")
    payload: dict[str, Any] = {
        "kind": "sign_autocorr_real",
        "schema": "sign_autocorr_real.v1",
        "ticker": ticker,
        "lags": list(LAGS),
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "power_law_sign_decay_measured_real_vs_sim",
        "interpretation": (
            "Lillo-Farmer: real sign autocorr decays as k^-gamma, "
            "gamma ~ 0.5. Splitting produces the mechanism; iid gives "
            "exponent ~ inf/nan (curve dead past lag 2)."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
