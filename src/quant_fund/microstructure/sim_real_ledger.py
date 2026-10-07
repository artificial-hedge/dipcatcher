"""sim_real_ledger — side-by-side sim-vs-real microstructure scorecard.

Runs identical measurements on the ZI-LOB sim arms and the real
LOBSTER tape and reports the honest divergence map: which stylized
facts our simulator reproduces (hump-shaped book, long-memory flow)
and by how much it misses. MIXED data_label — sim arms are SYNTHETIC,
the LOBSTER leg is a real NASDAQ tape.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.lobster import (
    EXECUTION,
    EXECUTION_HIDDEN,
    HALT,
    LobsterBook,
    parse_messages,
    parse_orderbook_row,
    resync_band,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SIM_REAL_SCHEMA = "sim_real_ledger.v1"


def _sign_lag1(signs: NDArray[np.float64]) -> float:
    if signs.size < 3:
        return float("nan")
    return float(np.dot(signs[:-1], signs[1:]) / signs.size)


def _hump_level(depth: list[float]) -> float | None:
    if len(depth) < 2:
        return None
    return float(int(np.argmax(depth)) + 1)


def measure_sim(
    config: ZILobConfig,
    flow: MOFlow | None,
    *,
    horizon: int = 20000,
) -> dict[str, Any]:
    """Event-granular stats from the ZI-LOB sim."""
    sim = ZILobSimulator(config, flow=flow)
    signs: list[float] = []
    spreads: list[float] = []
    depth_acc = np.zeros(5)
    n_depth = 0
    n_mo = 0
    mids: list[float] = []
    for _ in range(horizon):
        kind = sim.step()
        while len(sim.trades) > len(signs):
            tr = sim.trades[len(signs)]
            signs.append(1.0 if tr.aggressor == "buy" else -1.0)
        if kind == "market":
            n_mo += 1
        m = sim.mid
        if m is not None:
            mids.append(m)
        bb, ba = sim.best_bid, sim.best_ask
        if bb is not None and ba is not None:
            spreads.append(ba - bb)
        if len(depth_acc) == 5:
            try:
                depth_acc += np.array(
                    [sim.depth_at("buy", i + 1) + sim.depth_at("sell", i + 1) for i in range(5)],
                    dtype=np.float64,
                )
                n_depth += 1
            except KeyError:
                pass
    signs_a = np.asarray(signs)
    mids_a = np.asarray(mids)
    spreads_a = np.asarray(spreads)
    depth_mean = (depth_acc / max(1, n_depth)).tolist()
    mo_frac = n_mo / horizon
    return {
        "n_events": int(horizon),
        "n_trades": int(signs_a.size),
        "mo_fraction": float(mo_frac),
        "sign_lag1": _sign_lag1(signs_a),
        "spread_ticks_median": float(np.median(spreads_a) / config.tick)
        if spreads_a.size
        else float("nan"),
        "spread_ticks_p95": float(np.percentile(spreads_a, 95) / config.tick)
        if spreads_a.size
        else float("nan"),
        "mid_move_std": float(np.diff(mids_a).std() / config.tick)
        if mids_a.size > 1
        else float("nan"),
        "depth_l1_5_mean": depth_mean,
        "hump_level": _hump_level(depth_mean),
        "units": "ticks",
    }


def measure_lobster(msg: Path, ob: Path, *, tick_units: float = 100.0) -> dict[str, Any]:
    """Same metrics on the real tape (event granularity, resync-safe).

    tick_units: LOBSTER price units per tick (AMZN 2012: $0.01 × 10⁴ =
    100). Spreads and mid moves are expressed in ticks so they compare
    directly with the sim.
    """
    book = LobsterBook()
    signs: list[float] = []
    spreads: list[float] = []
    mids: list[float] = []
    depth_acc = np.zeros(5)
    n_depth = 0
    n_mo = 0
    seeded = False
    stride = 137
    n_events = 0
    with ob.open() as f:
        for i, (ev, row) in enumerate(zip(parse_messages(msg), csv.reader(f), strict=True)):
            ae, be = parse_orderbook_row(row)
            if not seeded:
                # Row 0 is the book state AFTER message 0: seeding from it
                # already includes event 0 — applying it would double-count
                # the first event.
                book.seed(ae, be)
                seeded = True
                n_events += 1
                if ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
                    n_mo += 1
                    signs.append(float(-ev.direction))
                continue
            book.apply(ev)
            n_events += 1
            if ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
                # MO prints include hidden-liquidity fills (type 5); record
                # before the no-visible-book skip so n_mo and the sign
                # stream cover the full aggressor tape.
                n_mo += 1
                signs.append(float(-ev.direction))
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", 10) != ae or book.top("bid", 10) != be:
                resync_band(book, ae, be)
            bb, ba = book.top("bid", 1), book.top("ask", 1)
            if bb and ba:
                spreads.append(float(ba[0][0] - bb[0][0]) / tick_units)
                mids.append((ba[0][0] + bb[0][0]) / (2 * tick_units))
            if i % stride == 0:
                top_b = book.top("bid", 5)
                top_a = book.top("ask", 5)
                for lvl in range(5):
                    depth_acc[lvl] += (top_b[lvl][1] if lvl < len(top_b) else 0) + (
                        top_a[lvl][1] if lvl < len(top_a) else 0
                    )
                n_depth += 1
    signs_a = np.asarray(signs)
    mids_a = np.asarray(mids)
    spreads_a = np.asarray(spreads)
    depth_mean = (depth_acc / max(1, n_depth)).tolist()
    return {
        "n_events": n_events,
        "n_trades": int(signs_a.size),
        "mo_fraction": float(n_mo / max(1, n_events)),
        "sign_lag1": _sign_lag1(signs_a),
        "spread_ticks_median": float(np.median(spreads_a)),
        "spread_ticks_p95": float(np.percentile(spreads_a, 95)),
        "mid_move_std": float(np.diff(mids_a).std()),
        "depth_l1_5_mean": depth_mean,
        "hump_level": _hump_level(depth_mean),
        "units": "ticks",
    }


def sim_real_ledger(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = next(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = next(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    real = measure_lobster(msg, ob)
    calm = measure_sim(ZILobConfig(seed=7), None, horizon=20000)
    regime = measure_sim(
        ZILobConfig(seed=7),
        MarkovRegimeFlow(
            states=(
                RegimeState("calm", 1.0, 0.5),
                RegimeState("trend", 2.2, 0.78),
            ),
            stay_probs=(0.97, 0.94),
            seed=11,
        ),
        horizon=20000,
    )
    metrics = (
        "sign_lag1",
        "mo_fraction",
        "spread_ticks_median",
        "mid_move_std",
    )
    table: dict[str, dict[str, float | None]] = {}
    for m in metrics:
        table[m] = {
            "sim_calm": calm.get(m),
            "sim_regime": regime.get(m),
            "real": real.get(m),
            "calm_over_real": (calm[m] / real[m] if real.get(m) not in (0, None) else None),
            "regime_over_real": (regime[m] / real[m] if real.get(m) not in (0, None) else None),
        }
    divergences = []
    if real["sign_lag1"] and regime["sign_lag1"] < real["sign_lag1"] - 0.05:
        divergences.append("sim_underestimates_sign_memory")
    if (
        calm["hump_level"] is not None
        and real["hump_level"] is not None
        and abs(calm["hump_level"] - real["hump_level"]) >= 1
    ):
        divergences.append(f"hump_level sim={calm['hump_level']} real={real['hump_level']}")
    payload: dict[str, Any] = {
        "schema": SIM_REAL_SCHEMA,
        "kind": "sim_real_ledger",
        "ticker": ticker,
        "table": table,
        "sim_calm": calm,
        "sim_regime": regime,
        "real": real,
        "divergences": divergences,
        "interpretation": (
            "Same event-granular measurements on ZI-LOB sim arms and the "
            "real LOBSTER tape. Ratios near 1 mean the sim reproduces "
            "the stylized fact; systematic gaps are named in "
            "`divergences` — calibration targets for the sim, not "
            "defects of the measurement"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
