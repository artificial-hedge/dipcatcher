"""mid_jump — distribution of mid-price moves per event, in ticks.

Every orderbook row change gives a new (bid,ask) midpoint. |Δmid| in
ticks measures move granularity: tight-book names move in 1-tick
steps; wide-book names take multi-tick jumps. The jump-size histogram
+ time between mid moves is a clean distributional fact the sim's
band-stepping book either reproduces or not.
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

JUMP_BINS = (0.0, 0.5, 1.5, 2.5, 4.5, 9.5, 20.5, 1 << 30)
JUMP_LABELS = ("0", "1", "2", "3-4", "5-9", "10-20", ">20")


def _jump_profile(times: list[float], mids_ticks: list[float]) -> dict[str, Any]:
    """Histogram |Δmid| over consecutive (t, mid) samples + move interarrival."""
    if len(mids_ticks) < 2:
        return {"ok": False, "reason": "too_few_samples"}
    moves = np.abs(np.diff(np.asarray(mids_ticks, dtype=float)))
    hist = np.zeros(len(JUMP_LABELS), dtype=float)
    for j in moves:
        for bi, (lo, hi) in enumerate(zip(JUMP_BINS[:-1], JUMP_BINS[1:], strict=True)):
            if lo <= j < hi:
                hist[bi] += 1
                break
    hist /= hist.sum()
    nonzero = moves[moves > 0]
    n_move = int(nonzero.size)
    move_times = [times[i + 1] for i, j in enumerate(moves) if j > 0]
    gaps = np.diff(np.asarray(move_times, dtype=float)) if len(move_times) > 1 else np.asarray([])
    return {
        "n_obs": int(moves.size),
        "n_mid_moves": n_move,
        "move_share": round(n_move / moves.size, 4),
        "mean_abs_jump_ticks": round(float(nonzero.mean()), 4) if n_move else 0.0,
        "p95_jump_ticks": round(float(np.percentile(nonzero, 95)), 4) if n_move else 0.0,
        "mean_move_gap_s": round(float(gaps.mean()), 4) if gaps.size else None,
        "jump_hist": {lab: round(float(hist[i]), 4) for i, lab in enumerate(JUMP_LABELS)},
    }


def lobster_mid_jumps(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    times: list[float] = []
    mids: list[float] = []
    with ob.open() as fo:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(fo), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                mids.append((asks[0][0] + bids[0][0]) / 200.0)  # raw→ticks
                times.append(ev.time_s)
    return _jump_profile(times, mids)


def sim_mid_jumps(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
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
    return _jump_profile(times, mids)


def mid_jump_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Jump-size comparison real vs sim arms. Sealed."""
    real = lobster_mid_jumps(tape_dir, ticker)
    arms = {
        "iid": sim_mid_jumps(seed=seed),
        "regime": sim_mid_jumps(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_mid_jumps(
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
    r_mean = real.get("mean_abs_jump_ticks")
    for name, arm in arms.items():
        a_mean = arm.get("mean_abs_jump_ticks")
        if r_mean and a_mean and abs(r_mean - a_mean) / r_mean > 0.5:
            divergences.append(f"{name}_jump_mean_ratio_{a_mean / r_mean:.2f}")
    payload: dict[str, Any] = {
        "kind": "mid_jump",
        "schema": "mid_jump.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "mid_jump_distribution_measured",
        "interpretation": (
            "Wide-book stocks take multi-tick mid steps; tight books "
            "rarely jump >1 tick. The histogram is a liquidity-regime "
            "fingerprint."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
