"""cancel_cluster — post-execution cancel retreat on real tape vs sim.

Hit the book and it flinches: after a market order sweeps a level,
resting liquidity gets repriced/canceled — resiliency is the measurable.
For each EXECUTION we count same-side CANCEL_PARTIAL+DELETE events in
the next `window_s` seconds and compare against that side's baseline
cancel rate. ratio > 1 = the book retreats after being hit.

Sim arms run the identical conditional-intensity estimator over the
sim's own event stream (we track submits/cancels/fills per step with
their continuous timestamps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
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

WINDOWS = (0.5, 2.0, 10.0, 60.0)


def _lift(exec_times: list[float], cxl_times: list[float], t0: float, t1: float) -> dict[str, Any]:
    """Post-exec cancel count per window vs side baseline rate."""
    e = np.asarray(exec_times)
    c = np.asarray(cxl_times)
    span = t1 - t0
    if e.size < 20 or span <= 0 or c.size < 20:
        return {"ok": False, "reason": "too_few_events"}
    base_rate = c.size / span
    out: dict[str, Any] = {"baseline_cxl_per_s": round(float(base_rate), 5)}
    for w in WINDOWS:
        # count cancels in (t_exec, t_exec + w] per exec
        idx = np.searchsorted(c, e, side="right")
        idx2 = np.searchsorted(c, e + w, side="right")
        counts = idx2 - idx
        counts = counts[e + w <= t1]  # censor execs near session end
        obs_rate = float(counts.mean() / w) if counts.size else None
        out[f"lift_{w:g}s"] = (
            round(obs_rate / base_rate, 4) if obs_rate is not None and base_rate > 0 else None
        )
        out[f"post_cxl_per_s_{w:g}"] = obs_rate
    return out


def lobster_cancel_cluster(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Conditional cancel intensity after execs, per side, on the tape."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    exec_t: dict[str, list[float]] = {"buy": [], "sell": []}
    cxl_t: dict[str, list[float]] = {"buy": [], "sell": []}
    t0 = t1 = 0.0
    for ev in parse_messages(msg):
        t1 = ev.time_s
        if t0 == 0.0:
            t0 = ev.time_s
        if ev.event_type == EXECUTION:
            # resting side hit: direction -1 => sell book lifted => sell cancels retreat
            side = "sell" if ev.direction == -1 else "buy"
            exec_t[side].append(ev.time_s)
        elif ev.event_type in (CANCEL_PARTIAL, DELETE):
            side = "sell" if ev.direction == -1 else "buy"
            cxl_t[side].append(ev.time_s)
    return {
        "n_execs": int(len(exec_t["buy"]) + len(exec_t["sell"])),
        "n_cancels": int(len(cxl_t["buy"]) + len(cxl_t["sell"])),
        "buy_side": _lift(exec_t["buy"], cxl_t["buy"], t0, t1),
        "sell_side": _lift(exec_t["sell"], cxl_t["sell"], t0, t1),
    }


def sim_cancel_cluster(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    """Same estimator on the sim: exec times + resting-order cancel times."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(horizon):
        sim.step()
    # The sim's anonymous theta_cxl cancels are not per-order timestamped,
    # so the honest sim-side analog is the exec-on-exec conditional lift
    # (post-trade clustering of the aggressive stream itself).
    all_exec = sorted(tr.t for tr in sim.trades)
    t0 = all_exec[0] if all_exec else 0.0
    t1 = sim.t
    out = {"n_execs": len(all_exec), "mode": "exec_on_exec_lift"}
    out["both_sides"] = _lift(all_exec, all_exec, t0, t1)
    return out


def cancel_cluster_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Cancel-retreat lift on the real tape vs sim arms. Sealed."""
    real = lobster_cancel_cluster(tape_dir, ticker)
    arms = {
        "iid": sim_cancel_cluster(seed=seed),
        "regime": sim_cancel_cluster(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_cancel_cluster(
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
    payload: dict[str, Any] = {
        "kind": "cancel_cluster",
        "schema": "cancel_cluster.v1",
        "ticker": ticker,
        "windows_s": list(WINDOWS),
        "real": real,
        "sim_arms": arms,
        "claim": "post_exec_cancel_retreat_measured",
        "interpretation": (
            "lift_k = post-exec cancel rate / baseline cancel rate, per "
            "hit side. >1 = book retreat (resiliency). The sim reports "
            "exec-on-exec lift instead (its anonymous theta_cxl cancels "
            "are not per-order timestamped) — the asymmetry is itself a "
            "documented mechanism gap."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
