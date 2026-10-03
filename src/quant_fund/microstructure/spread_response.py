"""spread_response — does a trade widen the book, and for how long?

The signature of informed flow on quotes: the conditional spread
response kernel. For each visible EXECUTION on the tape, spread
immediately before the event, then at horizons 10ms/50ms/100ms/500ms/
1s/5s after — split by aggressor direction, centered on the
event-day median spread. A positive kernel decaying back is the book
defending itself after impact; a permanent shift is repricing.

Same measurement on each ZI-LOB flow arm (which answers: does the
sim's book even react to fills — its spread is set by placement
dynamics, not by fear).
"""

from __future__ import annotations

import csv as _csv
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

HORIZONS_S = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)


def _response_kernel(
    ev_times: np.ndarray,
    ev_spreads: np.ndarray,
    exec_idx: np.ndarray,
    exec_dirs: np.ndarray,
) -> dict[str, Any]:
    """Median (spread@t+h − spread@t) per horizon, per direction."""
    baseline = float(np.median(ev_spreads)) if ev_spreads.size else None
    if baseline is None or exec_idx.size < 50:
        return {"ok": False, "n_execs": int(exec_idx.size)}
    out: dict[str, Any] = {"ok": True, "baseline_spread": baseline, "by_dir": {}}
    for d, name in ((1, "buy_initiated"), (-1, "sell_initiated"), (0, "pooled")):
        mask = exec_dirs == d if d else np.ones(exec_idx.size, dtype=bool)
        idx = exec_idx[mask]
        if idx.size < 20:
            out["by_dir"][name] = {"ok": False, "n": int(idx.size)}
            continue
        row: dict[str, Any] = {"ok": True, "n": int(idx.size), "kernel": {}}
        s0 = ev_spreads[idx]
        for h in HORIZONS_S:
            j = np.searchsorted(ev_times, ev_times[idx] + h, side="left")
            j = np.minimum(j, ev_spreads.size - 1)
            delta = ev_spreads[j] - s0
            row["kernel"][f"{h:g}s"] = {
                "median_delta": float(np.median(delta)),
                "mean_delta": float(delta.mean()),
                "share_wider": float(np.mean(delta > 0)),
            }
        out["by_dir"][name] = row
    return out


def lobster_spread_response(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    """Spread kernel around executions, using the orderbook snapshot
    file as ground truth: row i is the book state after event i, so the
    pre-event spread for event i is row i-1's spread."""
    times: list[float] = []
    spreads: list[int] = []
    exec_idx: list[int] = []
    exec_dirs: list[int] = []
    with ob_path.open() as f_ob:
        for i, (ev, ob_row) in enumerate(
            zip(parse_messages(msg_path), _csv.reader(f_ob), strict=True)
        ):
            asks, bids = parse_orderbook_row(ob_row)
            spread_now = asks[0][0] - bids[0][0] if asks and bids else -1
            times.append(ev.time_s)
            spreads.append(spread_now)
            if ev.event_type == EXECUTION:
                exec_idx.append(i)
                exec_dirs.append(-ev.direction)
    sp = np.asarray(spreads)
    valid = sp >= 0
    ev_times = np.asarray(times)[valid]
    sp = sp[valid]
    idx_map = np.cumsum(valid) - 1
    exec_idx_a = np.asarray(exec_idx)
    keep = valid[exec_idx_a] & (exec_idx_a > 0)
    exec_idx_a = idx_map[exec_idx_a[keep]]
    exec_dirs_a = np.asarray(exec_dirs)[keep]
    # s0 must be the PRE-event spread: step back one row in the valid index space
    exec_idx_a = np.maximum(exec_idx_a - 1, 0)
    return _response_kernel(ev_times, sp, exec_idx_a, exec_dirs_a)


def sim_spread_response(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    times: list[float] = []
    spreads: list[int] = []
    exec_t: list[float] = []
    exec_dirs: list[int] = []
    n_before = 0
    for _ in range(horizon):
        sim.step()
        times.append(sim.t)
        spreads.append(sim.spread_ticks if sim.spread_ticks is not None else 0)
        for tr in sim.trades[n_before:]:
            exec_t.append(tr.t)
            exec_dirs.append(1 if tr.aggressor == "buy" else -1)
        n_before = len(sim.trades)
    ev_times = np.asarray(times)
    sp = np.asarray(spreads)
    exec_idx = np.searchsorted(ev_times, np.asarray(exec_t), side="right") - 1
    return _response_kernel(ev_times, sp, exec_idx, np.asarray(exec_dirs))


def spread_response_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook CSV pair under {tape_dir}")
    real = lobster_spread_response(msg[0], ob[0])
    arms = {
        "iid": sim_spread_response(seed=seed),
        "regime": sim_spread_response(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_spread_response(
            flow=SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        rk = real["by_dir"].get("pooled", {}).get("kernel", {})
        for name, arm in arms.items():
            if not arm.get("ok"):
                continue
            ak = arm["by_dir"].get("pooled", {}).get("kernel", {})
            for h in HORIZONS_S:
                r = rk.get(f"{h:g}s", {}).get("median_delta")
                s = ak.get(f"{h:g}s", {}).get("median_delta")
                if r is not None and s is not None and abs(r - s) > 2.0:
                    divergences.append(f"{name}@{h}s_{s:.1f}_vs_{r:.1f}")
    payload: dict[str, Any] = {
        "kind": "spread_response",
        "schema": "spread_response.v1",
        "ticker": ticker,
        "horizons_s": list(HORIZONS_S),
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "spread_response_kernel_measured",
        "interpretation": (
            "kernel[h] = median (spread@t+h − spread@t) in ticks around "
            "executions, per aggressor direction. Positive-and-decaying = "
            "the book defends after impact; near-zero = fills barely move "
            "the quote state. share_wider = fraction of execs whose "
            "spread is wider at horizon h."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
