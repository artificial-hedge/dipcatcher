"""event_burst — burstiness of the full message stream.

Goh–Barabási burstiness B = (CV²-1)/(CV²+1) over inter-event times:
B=0 Poisson, B→1 maximally bursty, B<0 anti-bursty. Measured per event
type (submission/cancel/exec) AND the pooled stream — message traffic
on a real tape is strongly self-exciting even where execs alone are
spread thin (the waiting_times lane measured exec-only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    SUBMISSION,
    parse_messages,
)
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

_EVENT_NAMES = {
    SUBMISSION: "submission",
    CANCEL_PARTIAL: "cancel_partial",
    DELETE: "delete",
    EXECUTION: "execution",
    EXECUTION_HIDDEN: "execution_hidden",
}


def _burstiness(times: np.ndarray) -> dict[str, Any]:
    if times.size < 100:
        return {"ok": False, "n": int(times.size)}
    gaps = np.diff(times)
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size < 100 or gaps.mean() <= 0:
        return {"ok": False, "n": int(gaps.size)}
    cv = float(gaps.std(ddof=0) / gaps.mean())
    b = (cv * cv - 1.0) / (cv * cv + 1.0)
    return {
        "n_gaps": int(gaps.size),
        "mean_gap_ms": round(float(gaps.mean() * 1000.0), 4),
        "cv": round(cv, 4),
        "burstiness_B": round(b, 4),
        "p99_gap_ms": round(float(np.percentile(gaps, 99) * 1000.0), 4),
        "median_gap_ms": round(float(np.median(gaps) * 1000.0), 4),
    }


def lobster_burstiness(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    by_type: dict[str, list[float]] = {}
    all_t: list[float] = []
    for ev in parse_messages(msg):
        all_t.append(ev.time_s)
        name = _EVENT_NAMES.get(ev.event_type, str(ev.event_type))
        by_type.setdefault(name, []).append(ev.time_s)
    out: dict[str, Any] = {"all": _burstiness(np.asarray(all_t))}
    for name, ts in sorted(by_type.items()):
        out[name] = _burstiness(np.asarray(ts))
    return out


def sim_burstiness(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 30000,
    seed: int = 7,
) -> dict[str, Any]:
    """Sim has one event per step — pooled-stream B only."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    times = []
    for _ in range(horizon):
        sim.step()
        times.append(sim.t)
    return {"all": _burstiness(np.asarray(times))}


def burstiness_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Burstiness real vs sim pooled stream. Sealed."""
    real = lobster_burstiness(tape_dir, ticker)
    arms = {
        "iid": sim_burstiness(seed=seed),
        "regime": sim_burstiness(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_burstiness(
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
    rb = real.get("all", {}).get("burstiness_B")
    for name, arm in arms.items():
        ab = arm.get("all", {}).get("burstiness_B")
        if rb is not None and ab is not None and abs(ab - rb) > 0.15:
            divergences.append(f"{name}_B_{ab}_vs_{rb}")
    payload: dict[str, Any] = {
        "kind": "event_burst",
        "schema": "event_burst.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "burstiness_measured",
        "interpretation": (
            "B>0 = bursty (clustered activity); real message traffic is "
            "dominated by cancel/submit storms around execs."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
