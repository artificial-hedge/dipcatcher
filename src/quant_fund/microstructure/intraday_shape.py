"""intraday_shape — the U-shaped intraday activity profile.

The oldest intraday stylized fact (Wood, McInish & Ord 1985): volume,
volatility and spread all follow a U — heavy open, quiet lunch, heavy
close. The real tape lets us measure it directly: bin the session into
half-hour cells, count events/trades and mean quoted spread per cell.

Sim arms have no calendar — a flat intensity is expected; the divergence
field records per-cell ratios so a flow model that *does* inject a U
would show it honestly.
"""

from __future__ import annotations

import csv
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


def _u_shape_score(profile: np.ndarray) -> float | None:
    """(mean of edge thirds) / (mean of middle third) — >1 is U-shaped."""
    n = profile.size
    if n < 3:
        return None
    k = max(n // 3, 1)
    edge = profile[:k].mean() + profile[-k:].mean()
    mid = profile[k : n - k].mean()
    return float(edge / 2 / mid) if mid > 0 else None


def lobster_intraday(tape_dir: Path, ticker: str = "AMZN", *, n_bins: int = 13) -> dict[str, Any]:
    """Half-hour cells: events, execs, mean spread over the session."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    book = LobsterBook()
    t0, t1 = 34200.0, 57600.0
    edges = np.linspace(t0, t1, n_bins + 1)
    ev_counts = np.zeros(n_bins)
    ex_counts = np.zeros(n_bins)
    spread_sum = np.zeros(n_bins)
    spread_n = np.zeros(n_bins)
    seeded = False
    with ob.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(f_ob), strict=True):
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
            b = int(np.searchsorted(edges, ev.time_s, side="right") - 1)
            b = min(max(b, 0), n_bins - 1)
            ev_counts[b] += 1
            if ev.event_type == EXECUTION:
                ex_counts[b] += 1
            asks, bids = book.top("ask", 1), book.top("bid", 1)
            if asks and bids:
                spread_sum[b] += asks[0][0] - bids[0][0]
                spread_n[b] += 1
    mean_spread = (
        np.divide(spread_sum, spread_n, out=np.zeros_like(spread_sum), where=spread_n > 0) / 100.0
    )
    return {
        "n_bins": n_bins,
        "bin_seconds": float(edges[1] - edges[0]),
        "events_per_bin": [round(float(x)) for x in ev_counts],
        "execs_per_bin": [round(float(x)) for x in ex_counts],
        "mean_spread_per_bin": [round(float(x), 4) for x in mean_spread],
        "activity_u_shape": _u_shape_score(ex_counts if ex_counts.sum() else ev_counts),
        "spread_u_shape": _u_shape_score(mean_spread[mean_spread > 0]),
    }


def sim_intraday(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    n_bins: int = 13,
    seed: int = 7,
) -> dict[str, Any]:
    """Event-rate and spread profile in event-index bins (no calendar)."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    n_events = np.zeros(n_bins)
    spread_sum = np.zeros(n_bins)
    spread_n = np.zeros(n_bins)
    per = horizon / n_bins
    for i in range(horizon):
        sim.step()
        b = min(int(i / per), n_bins - 1)
        n_events[b] += 1
        s = sim.spread_ticks
        if s is not None:
            spread_sum[b] += s
            spread_n[b] += 1
    mean_spread = np.divide(spread_sum, spread_n, out=np.zeros_like(spread_sum), where=spread_n > 0)
    return {
        "activity_u_shape": _u_shape_score(n_events),
        "spread_u_shape": _u_shape_score(mean_spread[mean_spread > 0]),
        "mean_spread_ticks": round(float(mean_spread.mean()), 3),
    }


def intraday_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """U-shape measurement on the real session vs sim arms. Sealed."""
    real = lobster_intraday(tape_dir, ticker)
    arms = {
        "iid": sim_intraday(seed=seed),
        "regime": sim_intraday(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_intraday(
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
        "kind": "intraday_shape",
        "schema": "intraday_shape.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "claim": "intraday_u_shape_measured_real_vs_flat_sim",
        "interpretation": (
            "u_shape = edge-third mean / middle-third mean of activity. "
            "Real equities show >1 (U). Sim arms are flat by construction "
            "(~1.0) — a calendar-intensity term is an open calibration gap."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
