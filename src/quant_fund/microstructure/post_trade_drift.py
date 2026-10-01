"""post_trade_drift — does impact continue or revert after a fill?

The micro-scale complement to the propagator curve: for each
execution, the signed mid move at event-horizons {1, 5, 20, 50, 200}
events after. Positive kernel = continuation (the move is permanent
or accelerating); negative = reversion (transient impact being
absorbed back). Split by aggressor direction and by whether the exec
itself moved the mid — the conditional shape separates price-setting
fills from noise fills.

Same on the ZI-LOB sim (trade events → same kernel, event-count
horizons).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import EXECUTION, parse_messages, parse_orderbook_row
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

EVENT_HORIZONS = (1, 5, 20, 50, 200)


def _drift_kernel(
    mids: np.ndarray,
    exec_idx: np.ndarray,
    exec_dirs: np.ndarray,
) -> dict[str, Any]:
    if exec_idx.size < 50 or mids.size < 2:
        return {"ok": False, "n_execs": int(exec_idx.size)}
    instant = exec_dirs * (mids[exec_idx] - mids[np.maximum(exec_idx - 1, 0)])
    out: dict[str, Any] = {"ok": True, "n_execs": int(exec_idx.size)}
    for name, mask in (
        ("all", np.ones(exec_idx.size, dtype=bool)),
        ("movers", instant != 0),
        ("nonmovers", instant == 0),
    ):
        idx = exec_idx[mask]
        dirs = exec_dirs[mask]
        if idx.size < 20:
            out[name] = {"ok": False, "n": int(idx.size)}
            continue
        row: dict[str, Any] = {
            "ok": True,
            "n": int(idx.size),
            "instant_mean_ticks": float((dirs * instant[mask]).mean()),
            "kernel": {},
        }
        for h in EVENT_HORIZONS:
            j = np.minimum(idx + h, mids.size - 1)
            drift = dirs * (mids[j] - mids[idx])
            row["kernel"][str(h)] = {
                "mean_ticks": float(drift.mean()),
                "median_ticks": float(np.median(drift)),
                "share_positive": float(np.mean(drift > 0)),
            }
        out[name] = row
    return out


def lobster_post_trade_drift(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    mids: list[float] = []
    exec_idx: list[int] = []
    exec_dirs: list[int] = []
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                mids.append((asks[0][0] + bids[0][0]) / 200.0)
            elif mids:
                mids.append(mids[-1])
            else:
                continue
            if ev.event_type == EXECUTION:
                exec_idx.append(len(mids) - 1)
                exec_dirs.append(-ev.direction)
    return _drift_kernel(np.asarray(mids), np.asarray(exec_idx), np.asarray(exec_dirs))


def sim_post_trade_drift(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    mids: list[float] = []
    exec_idx: list[int] = []
    exec_dirs: list[int] = []
    n_before = 0
    for _ in range(horizon):
        sim.step()
        mid = sim.mid if sim.mid is not None else (mids[-1] if mids else 0.0)
        mids.append(mid)
        for tr in sim.trades[n_before:]:
            exec_idx.append(len(mids) - 1)
            exec_dirs.append(1 if tr.aggressor == "buy" else -1)
        n_before = len(sim.trades)
    return _drift_kernel(np.asarray(mids), np.asarray(exec_idx), np.asarray(exec_dirs))


def post_trade_drift_bench(
    tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7
) -> dict[str, Any]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook CSV pair under {tape_dir}")
    real = lobster_post_trade_drift(msg[0], ob[0])
    arms = {
        "iid": sim_post_trade_drift(seed=seed),
        "regime": sim_post_trade_drift(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_post_trade_drift(
            flow=SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        rk = real.get("all", {}).get("kernel", {})
        for name, arm in arms.items():
            if not arm.get("ok"):
                continue
            ak = arm.get("all", {}).get("kernel", {})
            for h in EVENT_HORIZONS:
                r = rk.get(str(h), {}).get("mean_ticks")
                s = ak.get(str(h), {}).get("mean_ticks")
                if r is not None and s is not None and abs(r - s) > 0.5:
                    divergences.append(f"{name}@{h}ev_{s:.2f}_vs_{r:.2f}")
    payload: dict[str, Any] = {
        "kind": "post_trade_drift",
        "schema": "post_trade_drift.v1",
        "ticker": ticker,
        "event_horizons": list(EVENT_HORIZONS),
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "post_trade_drift_kernel_measured",
        "interpretation": (
            "kernel[h] = mean aggressor-signed Δmid (ticks) from the exec "
            "to h events later. Movers vs nonmovers split fills that did "
            "vs did not shift the mid at the event. Positive = impact "
            "continues; negative = the touch reverts."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
