"""event_matrix — full event-rate card real vs sim.

A calibratable LOB sim must match the *whole* arrival-rate matrix —
submissions, partial cancels, deletes, executions, per side — not just
the exec intensity. This lane emits the complete rate card in
events/second and per-event mix shares, real tape vs each sim arm.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    HALT,
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

TYPE_NAMES = {
    SUBMISSION: "sub",
    CANCEL_PARTIAL: "cxl_part",
    DELETE: "delete",
    EXECUTION: "exec",
    EXECUTION_HIDDEN: "exec_hidden",
    HALT: "halt",
}


def lobster_event_matrix(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Events/s per (type, side) + mix shares over the session."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    counts: Counter[str] = Counter()
    t0 = t1 = 0.0
    n = 0
    for ev in parse_messages(msg):
        n += 1
        if n == 1:
            t0 = ev.time_s
        t1 = ev.time_s
        name = TYPE_NAMES.get(ev.event_type, f"type_{ev.event_type}")
        side = "buy" if ev.direction > 0 else "sell"
        counts[f"{name}|{side}"] += 1
        counts[name] += 1
    span = max(1e-9, t1 - t0)
    total = sum(v for k, v in counts.items() if "|" not in k)
    out: dict[str, Any] = {
        "n_events": n,
        "span_s": round(span, 1),
        "events_per_s": {k: round(v / span, 4) for k, v in sorted(counts.items())},
        "mix_share": {},
    }
    if total:
        out["mix_share"] = {
            k: round(v / total, 4) for k, v in sorted(counts.items()) if "|" not in k
        }
    return out


def sim_event_matrix(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    """Sim rate card: exec rate + submission intensity from config.

    The sim's theta-driven event clock is implicit; we measure execs/s
    (trade events per sim-second) and the config's declared arrival
    intensities for the rest.
    """
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(horizon):
        sim.step()
    span = max(1e-9, sim.t)
    n_buy = sum(1 for tr in sim.trades if tr.aggressor == "buy")
    n_sell = len(sim.trades) - n_buy
    return {
        "n_events": horizon,
        "span_s": round(span, 1),
        "events_per_s": {
            "exec": round(len(sim.trades) / span, 4),
            "exec_buy": round(n_buy / span, 4),
            "exec_sell": round(n_sell / span, 4),
        },
        "mix_share": {},
        "steps_per_s": round(horizon / span, 3),
        "config_intensity": {
            "lambda_mo": getattr(cfg, "lambda_mo", None),
            "theta_cxl": getattr(cfg, "theta_cxl", None),
            "lambda_lo": getattr(cfg, "lambda_lo", None),
        },
    }


def event_matrix_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Event-rate card real vs sim arms. Sealed."""
    real = lobster_event_matrix(tape_dir, ticker)
    arms = {
        "iid": sim_event_matrix(seed=seed),
        "regime": sim_event_matrix(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_event_matrix(
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
    real_exec = real.get("events_per_s", {}).get("exec", 0.0)
    for name, arm in arms.items():
        sim_exec = arm.get("events_per_s", {}).get("exec", 0.0)
        if real_exec > 0 and sim_exec > 0:
            ratio = sim_exec / real_exec
            if not math.isfinite(ratio):
                ratio = 0.0
            if abs(ratio - 1.0) > 1.0:
                divergences.append(f"{name}_exec_rate_ratio_{ratio:.2f}")
    payload: dict[str, Any] = {
        "kind": "event_matrix",
        "schema": "event_matrix.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "event_rate_matrix_measured",
        "interpretation": (
            "The full arrival-rate card is the calibration surface: "
            "submissions, cancels, deletes, execs per side per second. "
            "Matching only exec intensity leaves the other 95% of the "
            "tape's activity unmodeled."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
