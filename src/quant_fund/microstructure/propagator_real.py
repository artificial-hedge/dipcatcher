"""propagator_real — the price response function on real tape and sim.

The propagator R(l) = E[eps_i * (m_{i+l} - m_i)] is the expected signed
mid move l events after a market order with sign eps_i. It is the
empirical footprint of impact: how much a trade moves the price over the
following events. Bouchaud's transient-impact model has the bare
per-trade impact decaying like l^{-gamma} while the cumulative response
can still rise concavely; that offset against persistent order flow is
what makes returns nearly uncorrelated despite long sign memory. The
fitted log-log slope is reported sign-honestly: a positive exponent
means the response keeps accumulating through the horizon.

The estimator is identical on both sides: mids indexed by book event,
trade signs indexed by the event that produced them, response averaged
over trades whose horizon fits inside the stream. Sim arms run the same
measurement on the ZI-LOB under {iid, regime, split} flow, so the gap to
the real curve is attributable to the flow mechanism, not the estimator.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
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
from quant_fund.microstructure.maker_age_bench import _MO_PMF, _spec
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _propagator(mids: list[float], trade_idx: list[int], signs: list[int]) -> dict[str, Any]:
    """R(l) in ticks + a log-log decay fit over positive responses."""
    m = np.asarray(mids, dtype=float)
    idx = np.asarray(trade_idx, dtype=int)
    eps = np.asarray(signs, dtype=float)
    n = m.size

    def _f(v: float | None) -> float | None:
        return v if v is not None and math.isfinite(v) else None

    out: dict[str, float | None] = {}
    for lag in LAGS:
        ok = idx + lag < n
        if int(ok.sum()) < 5:
            out[str(lag)] = None
            continue
        resp = eps[ok] * (m[idx[ok] + lag] - m[idx[ok]])
        out[str(lag)] = float(resp.mean())
    lags = np.asarray([lag for lag in LAGS if out[str(lag)] is not None], dtype=float)
    vals = np.asarray([out[str(int(lag))] for lag in lags], dtype=float)
    pos = vals > 0
    loglog_exponent: float | None = None
    fit_log_intercept: float | None = None
    if int(pos.sum()) >= 3:
        slope, intercept = np.polyfit(np.log(lags[pos]), np.log(vals[pos]), 1)
        loglog_exponent = float(-slope)
        fit_log_intercept = float(intercept)
    return {
        "n_trades": int(idx.size),
        "n_events": int(n),
        "response_ticks": {k: _f(round(v, 6)) if v is not None else None for k, v in out.items()},
        "loglog_exponent": _f(loglog_exponent) if loglog_exponent is not None else None,
        "fit_log_intercept": _f(fit_log_intercept) if fit_log_intercept is not None else None,
    }


def propagator_lobster(
    message_path: Path, orderbook_path: Path, *, tick_units: float = 100.0
) -> dict[str, Any]:
    """Replay the real tape, collecting mid + trade sign per event index."""
    book = LobsterBook()
    mids: list[float] = []
    trade_idx: list[int] = []
    signs: list[int] = []
    seeded = False
    with orderbook_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(message_path), csv.reader(f_ob), strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if not seeded:
                book.seed(asks_exp, bids_exp)
                book.apply(ev)
                seeded = True
                continue
            book.apply(ev)
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", 10) != asks_exp or book.top("bid", 10) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
            asks, bids = book.top("ask", 1), book.top("bid", 1)
            if not asks or not bids:
                continue
            mids.append((asks[0][0] + bids[0][0]) / (2.0 * tick_units))
            if ev.event_type == EXECUTION:
                trade_idx.append(len(mids) - 1)
                signs.append(-ev.direction)  # exec dir = resting side
    return _propagator(mids, trade_idx, signs)


def propagator_sim(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    """Same estimator on the sim: mid + trade sign indexed by event step."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    mids: list[float] = []
    trade_idx: list[int] = []
    signs: list[int] = []
    n_seen = 0
    for _ in range(horizon):
        sim.step()
        new = sim.trades[n_seen:]
        n_seen = len(sim.trades)
        mid = sim.mid
        if mid is None:
            continue
        mids.append(mid / cfg.tick)  # tick units, same as the tape
        if new:
            trade_idx.append(len(mids) - 1)
            signs.append(1 if new[0].aggressor == "buy" else -1)
    return _propagator(mids, trade_idx, signs)


def propagator_real_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Real-tape propagator vs the three sim flow arms. Sealed receipt."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    real = propagator_lobster(msg, ob)
    regime = MarkovRegimeFlow(
        states=(
            RegimeState("calm", 1.0, 0.5),
            RegimeState("bursty", 3.0, 0.62),
        ),
        stay_probs=(0.995, 0.985),
        seed=seed,
    )
    split = SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed
    )
    deep = replace(
        santa_fe_config(seed=seed + 3),
        hawkes=_spec(),
        lo_offset_gain=80.0,
        touch_pull=0.4,
        cxl_touch_bias=0.5,
        cxl_dist_decay=3.0,
        cxl_requote=0.5,
        mo_size_pmf=_MO_PMF,
        band=40,
        lam=3.5,
        theta_cxl=0.4,
        lo_offset=4,
    )
    split_deep = SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 3
    )
    arms = {
        "iid": propagator_sim(seed=seed),
        "regime": propagator_sim(flow=regime, seed=seed + 1),
        "split": propagator_sim(flow=split, seed=seed + 2),
        "deep_split": propagator_sim(deep, flow=split_deep, seed=seed + 3, horizon=60000),
    }
    table: dict[str, Any] = {"real": real, "sim_arms": arms}
    divergences: list[str] = []
    r1 = real["response_ticks"].get("1")
    for arm, res in arms.items():
        s1 = res["response_ticks"].get("1")
        if r1 is not None and s1 is not None and abs(s1) > 1e-9:
            table[f"ratio_real_over_{arm}@1"] = round(r1 / s1, 4)
        if r1 is None or s1 is None or abs(r1 - s1) > 0.5 * max(abs(r1), 0.05):
            divergences.append(f"{arm}_lag1_response_off")
    payload: dict[str, Any] = {
        "kind": "propagator_real",
        "schema": "propagator_real.v1",
        "ticker": ticker,
        "lags_events": list(LAGS),
        "estimator": "R(l) = E[sign_i * (mid_{i+l} - mid_i)] over book-event index; tick units",
        "table": table,
        "divergences": divergences,
        "claim": "propagator_measured_on_real_tape_vs_sim_arms",
        "interpretation": (
            "Immediate response R(1) is the effective spread-paid-per-"
            "trade; the log-log slope marks whether impact accumulates "
            "(positive) or is absorbed (negative) over the horizon. "
            "Sim arms share the estimator, so gaps isolate the flow "
            "mechanism, not the measurement. The deep_split arm pairs "
            "metaorder splitting with the deep-book regime (band=40, "
            "churn knobs) — testing whether book depth, not just flow "
            "correlation, closes the lag-1 undershoot."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
