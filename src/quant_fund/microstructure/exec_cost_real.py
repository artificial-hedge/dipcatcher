"""exec_cost_real — per-fill slippage beyond the mid on real tape vs sim.

Every execution pays the half-spread plus whatever depth it walks. The
measurable: `ticks_beyond_mid` = signed distance from pre-trade mid to
fill price, in ticks. Bucketed by trade size this is the raw material
of the square-root impact law — a buy exec that lifts two levels costs
more ticks than one resting at the touch.

On the sim the identical metric uses `tr.price` vs the mid at the step
the trade fired.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    EXECUTION,
    EXECUTION_HIDDEN,
    HALT,
    LobsterBook,
    parse_messages,
    parse_orderbook_row,
    resync_band,
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

SIZE_BUCKETS = ((1, 100), (101, 300), (301, 1000), (1001, 5000), (5001, 1 << 30))


def _bucket_stats(qty: np.ndarray, cost_ticks: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for lo, hi in SIZE_BUCKETS:
        m = (qty >= lo) & (qty < hi)
        if m.sum() < 3:
            continue
        out[f"qty_{lo}_{hi if hi < (1 << 30) else 'inf'}"] = {
            "n": int(m.sum()),
            "mean_ticks_beyond_mid": round(float(cost_ticks[m].mean()), 4),
            "p90_ticks": round(float(np.percentile(cost_ticks[m], 90)), 4),
        }
    return out


def _size_cost_exponent(qty: np.ndarray, cost: np.ndarray) -> float | None:
    """log-log slope of ticks-beyond-mid vs qty — pre-sqrt-law shape."""
    m = (qty > 0) & np.isfinite(cost) & (cost > 0)
    if m.sum() < 30:
        return None
    lx, ly = np.log(qty[m]), np.log(cost[m])
    if np.std(lx) < 1e-9 or np.std(ly) < 1e-9:
        return None
    try:
        slope = float(np.polyfit(lx, ly, 1)[0])
    except np.linalg.LinAlgError:
        return None
    return round(slope, 4) if math.isfinite(slope) else None


def lobster_exec_cost(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """Per-fill ticks-beyond-mid on the real tape."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    book = LobsterBook()
    qty: list[float] = []
    cost: list[float] = []
    seeded = False
    with ob.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(f_ob), strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if not seeded:
                book.seed(asks_exp, bids_exp)
                book.apply(ev)
                seeded = True
                continue
            # mid BEFORE the event eats liquidity
            asks, bids = book.top("ask", 1), book.top("bid", 1)
            pre_mid = (asks[0][0] + bids[0][0]) / 2.0 if asks and bids else None
            book.apply(ev)
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", 10) != asks_exp or book.top("bid", 10) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
            if ev.event_type == EXECUTION and pre_mid is not None:
                # direction is the resting side: -1 resting sell = buyer-initiated;
                # price is in 0.0001-dollar units -> /100 lands on ticks
                signed = -ev.direction * (ev.price - pre_mid) / 100.0
                qty.append(float(ev.size))
                cost.append(signed)
    q = np.asarray(qty)
    c = np.asarray(cost)
    out: dict[str, Any] = {
        "n_execs": int(q.size),
        "mean_ticks_beyond_mid": round(float(c.mean()), 4) if c.size else None,
        "median_ticks": round(float(np.median(c)), 4) if c.size else None,
        "share_negative_fills": round(float((c < 0).mean()), 4) if c.size else None,
        "size_buckets": _bucket_stats(q, c),
        "size_cost_exponent": _size_cost_exponent(q, c),
    }
    return out


def sim_exec_cost(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    """Ticks-beyond-mid for sim trades: tr.price vs the pre-step mid."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    qty: list[float] = []
    cost: list[float] = []
    n_before = 0
    for _ in range(horizon):
        pre_mid = sim.mid
        sim.step()
        for tr in sim.trades[n_before:]:
            if pre_mid is None:
                continue
            side = 1 if tr.aggressor == "buy" else -1
            cost.append(side * (tr.price - pre_mid) / cfg.tick)
            qty.append(float(tr.qty))
        n_before = len(sim.trades)
    q = np.asarray(qty)
    c = np.asarray(cost)
    return {
        "n_execs": int(q.size),
        "mean_ticks_beyond_mid": round(float(c.mean()), 4) if c.size else None,
        "median_ticks": round(float(np.median(c)), 4) if c.size else None,
        "share_negative_fills": round(float((c < 0).mean()), 4) if c.size else None,
        "size_buckets": _bucket_stats(q, c),
        "size_cost_exponent": _size_cost_exponent(q, c),
    }


def exec_cost_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Real vs sim per-fill cost. Sealed receipt."""
    real = lobster_exec_cost(tape_dir, ticker)
    arms = {
        "iid": sim_exec_cost(seed=seed),
        "regime": sim_exec_cost(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_exec_cost(
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
        "kind": "exec_cost_real",
        "schema": "exec_cost_real.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "claim": "per_fill_slippage_measured_real_vs_sim",
        "interpretation": (
            "ticks_beyond_mid > 0 pays the touch; < 0 means the fill was "
            "inside the mid (size dependence walks the book). The size-cost "
            "exponent is the local pre-cursor of the sqrt impact law — "
            "a deep-book sim and the real tape will not agree in buckets."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
