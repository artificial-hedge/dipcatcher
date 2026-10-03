"""Side imbalance: systematic buy-vs-sell asymmetry on the tape.

Four asymmetry channels measured identically on real LOBSTER tape and
the ZI-LOB sim:

- ``exec_buy_share``: fraction of visible EXECUTIONs whose resting side
  is sell (i.e. buy-initiated trades) — a drift in buy pressure.
- ``submission_buy_share``: fraction of SUBMISSIONs on the buy side.
- ``depth_imbalance``: time-average of (bid_sz1 − ask_sz1)/(bid+ask)
  at the touch — which side holds more resting depth.
- ``exec_size_asym``: mean buy-initiated vs sell-initiated fill size.

A market-wide up-drift should show exec_buy_share > 0.5 co-moving with
positive depth imbalance on the ask side (leaning into demand). The
sim's flows are side-symmetric by construction — any real-side skew is
a structural fact the ZI arms cannot express.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    EXECUTION,
    SUBMISSION,
    parse_messages,
    parse_orderbook_row,
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

N_HOUR_BINS = 8  # 09:30–16:00 session


def _imbalance_stats(
    exec_dirs: np.ndarray,
    exec_sizes: np.ndarray,
    exec_times: np.ndarray,
    sub_dirs: np.ndarray,
    touch_imb: np.ndarray,
    session_lo: float,
    session_span: float,
) -> dict[str, Any]:
    n_exec = exec_dirs.size
    if n_exec < 50:
        return {"ok": False, "n": int(n_exec)}
    buy = exec_dirs == 1
    per_bin: list[dict[str, Any]] = []
    for i in range(N_HOUR_BINS):
        lo = session_lo + session_span * i / N_HOUR_BINS
        hi = session_lo + session_span * (i + 1) / N_HOUR_BINS
        m = (exec_times >= lo) & (exec_times < hi)
        if m.sum() >= 10:
            per_bin.append({"bin": i, "n": int(m.sum()), "buy_share": float(buy[m].mean())})
    imb_ok = np.isfinite(touch_imb)
    return {
        "ok": True,
        "n_exec": int(n_exec),
        "exec_buy_share": float(buy.mean()),
        "exec_buy_share_per_bin": per_bin,
        "exec_size_buy_mean": float(exec_sizes[buy].mean()) if buy.any() else None,
        "exec_size_sell_mean": float(exec_sizes[~buy].mean()) if (~buy).any() else None,
        "submission_buy_share": float((sub_dirs == 1).mean()) if sub_dirs.size else None,
        "touch_imbalance_mean": float(np.mean(touch_imb[imb_ok])) if imb_ok.any() else None,
        "touch_imbalance_pos_share": float((touch_imb[imb_ok] > 0).mean())
        if imb_ok.any()
        else None,
    }


def lobster_side_imbalance(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    exec_dirs: list[int] = []
    exec_sizes: list[int] = []
    exec_times: list[float] = []
    sub_dirs: list[int] = []
    touch_imb: list[float] = []
    t_lo: float | None = None
    t_hi = 0.0
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            if t_lo is None:
                t_lo = ev.time_s
            t_hi = ev.time_s
            if ev.event_type == EXECUTION:
                # direction = resting side; aggressor is the opposite
                exec_dirs.append(-ev.direction)
                exec_sizes.append(ev.size)
                exec_times.append(ev.time_s)
            elif ev.event_type == SUBMISSION:
                sub_dirs.append(ev.direction)
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                a_sz, b_sz = float(asks[0][1]), float(bids[0][1])
                touch_imb.append((b_sz - a_sz) / (a_sz + b_sz))
            else:
                touch_imb.append(np.nan)
    return _imbalance_stats(
        np.asarray(exec_dirs),
        np.asarray(exec_sizes, dtype=float),
        np.asarray(exec_times),
        np.asarray(sub_dirs),
        np.asarray(touch_imb),
        t_lo or 0.0,
        (t_hi - (t_lo or 0.0)) or 1.0,
    )


def sim_side_imbalance(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    n0 = len(sim.trades)
    touch_imb: list[float] = []
    for _ in range(horizon):
        sim.step()
        a_sz = (
            float(sim.depth_at("sell", sim.best_ask_level))
            if sim.best_ask_level is not None
            else 0.0
        )
        b_sz = (
            float(sim.depth_at("buy", sim.best_bid_level))
            if sim.best_bid_level is not None
            else 0.0
        )
        touch_imb.append((b_sz - a_sz) / (a_sz + b_sz) if (a_sz + b_sz) > 0 else np.nan)
    trades = sim.trades[n0:]
    exec_dirs = np.asarray([1 if tr.aggressor == "buy" else -1 for tr in trades])
    exec_sizes = np.asarray([tr.qty for tr in trades], dtype=float)
    exec_times = np.asarray([tr.t for tr in trades])
    # submission side share from the live order registry
    sides = [o.side for o in sim._orders.values()]
    sub_dirs = np.asarray([1 if s == "buy" else -1 for s in sides])
    t_lo = float(exec_times.min()) if exec_times.size else 0.0
    t_hi = float(exec_times.max()) if exec_times.size else 1.0
    return _imbalance_stats(
        exec_dirs,
        exec_sizes,
        exec_times,
        sub_dirs,
        np.asarray(touch_imb),
        t_lo,
        (t_hi - t_lo) or 1.0,
    )


def side_imbalance_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_side_imbalance(msg_path, ob_path)
    arms = {
        "iid": sim_side_imbalance(seed=seed),
        "regime": sim_side_imbalance(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_side_imbalance(
            SplitFlow(
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
    divergences: list[str] = []
    if real.get("ok"):
        for name, arm in arms.items():
            if arm.get("ok") and abs(arm["exec_buy_share"] - real["exec_buy_share"]) > 0.05:
                divergences.append(
                    f"{name}_buyshare_{arm['exec_buy_share']:.3f}_vs_{real['exec_buy_share']:.3f}"
                )
    payload: dict[str, Any] = {
        "kind": "side_imbalance",
        "schema": "side_imbalance.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "side_asymmetry_measured",
        "interpretation": (
            "exec_buy_share = share of fills initiated by buyers (resting "
            "side == sell). submission_buy_share from SUBMISSIONs. "
            "touch_imbalance = time-mean of (bid_sz1 - ask_sz1)/(sum) at "
            "the touch. Per-bin buy share shows intraday drift in "
            "initiative. The ZI-LOB flows are side-symmetric — a large "
            "real skew is structural, not a calibration miss."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
