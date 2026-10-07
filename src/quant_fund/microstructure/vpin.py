"""vpin — volume-synchronized probability of informed trading.

Easley, López de Prado & O'Hara (2012): form equal-volume buckets, take
signed buy/sell volume per bucket, and VPIN is the mean absolute
order-flow imbalance over the last n buckets — a proxy for how one-sided
("toxic") flow has become. On LOBSTER the aggressor flag gives exact
trade signing (no Lee-Ready needed). The claim to measure honestly: does
the VPIN series actually precede volatility — corr(VPIN_t, |Δmid| over
the next bucket window)?

Sim arms use TradeEvent.aggressor on the ZI-LOB tape, identically
bucketed by equal volume.
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


def _bucket_imbalances(
    signed_qty: np.ndarray, bucket_volume: float
) -> tuple[np.ndarray, np.ndarray]:
    """Consume a signed-quantity stream into equal-volume buckets.

    Returns (bucket_imbalance, bucket_start_index): imbalance is
    |sum of signed qty| per bucket; start index marks the position in the
    trade stream where each bucket began (for forward-returns alignment).
    """
    out: list[float] = []
    starts: list[int] = []
    acc = 0.0
    vol = 0.0
    start = 0
    for i, q in enumerate(signed_qty):
        if vol == 0.0:
            start = i
        acc += q
        vol += abs(q)
        if vol >= bucket_volume:
            out.append(abs(acc))
            starts.append(start)
            acc = 0.0
            vol = 0.0
    return np.asarray(out), np.asarray(starts)


def _vpin_series(imbalances: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean imbalance over the last `window` buckets (unnormalized)."""
    if imbalances.size < window:
        return np.asarray([])
    cs = np.concatenate([[0.0], np.cumsum(imbalances)])
    return np.asarray((cs[window:] - cs[:-window]) / window)


def _vpin_stats(
    signed_qty: np.ndarray,
    mid_at_trade: np.ndarray,
    *,
    n_buckets: int = 50,
    window: int = 10,
) -> dict[str, Any]:
    """VPIN series + forward-vol correlation for one signed stream."""
    if signed_qty.size < 200:
        return {"ok": False, "reason": "few_trades"}
    bucket_volume = float(np.abs(signed_qty).sum() / n_buckets)
    imb, starts = _bucket_imbalances(signed_qty, bucket_volume)
    vpin = _vpin_series(imb, window) / bucket_volume
    # forward |mid change| across the first bucket strictly AFTER the VPIN
    # window — indexing starts[:len(vpin)] would overlap the predictor's own
    # window and report a contemporaneous, not forward, correlation
    fwd: list[float] = []
    for j in range(len(vpin)):
        b = j + window
        s = int(starts[b]) if b < starts.size else len(mid_at_trade) - 1
        nxt = int(starts[b + 1]) if b + 1 < starts.size else len(mid_at_trade) - 1
        if nxt > s and mid_at_trade[s] > 0 and mid_at_trade[nxt] > 0:
            fwd.append(abs(math.log(mid_at_trade[nxt] / mid_at_trade[s])))
        else:
            fwd.append(float("nan"))
    fwd_arr = np.asarray(fwd[: len(vpin)])
    mask = np.isfinite(fwd_arr)
    corr = (
        float(np.corrcoef(vpin[mask], fwd_arr[mask])[0, 1])
        if mask.sum() > 5 and np.std(vpin[mask]) > 0 and np.std(fwd_arr[mask]) > 0
        else None
    )
    return {
        "ok": True,
        "n_buckets": int(imb.size),
        "bucket_volume": round(bucket_volume, 2),
        "vpin_mean": round(float(vpin.mean()), 4) if vpin.size else None,
        "vpin_p90": round(float(np.percentile(vpin, 90)), 4) if vpin.size else None,
        "vpin_fwd_vol_corr": round(corr, 4) if corr is not None else None,
    }


def vpin_lobster(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    """VPIN on the real tape: signed execs + mid path for fwd vol."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    book = LobsterBook()
    signed: list[float] = []
    mids: list[float] = []
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
            asks, bids = book.top("ask", 1), book.top("bid", 1)
            if ev.event_type == EXECUTION and asks and bids:
                mids.append((asks[0][0] + bids[0][0]) / 200.0)
                signed.append(-ev.direction * ev.size)
    return _vpin_stats(np.asarray(signed, dtype=float), np.asarray(mids))


def vpin_sim(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    for _ in range(horizon):
        sim.step()
    signed = np.asarray(
        [tr.qty * (1 if tr.aggressor == "buy" else -1) for tr in sim.trades],
        dtype=float,
    )
    mids = np.asarray([float(tr.price) / cfg.tick for tr in sim.trades], dtype=float)
    return _vpin_stats(signed, mids)


def vpin_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """VPIN toxicity on real tape vs sim arms. Sealed receipt."""
    real = vpin_lobster(tape_dir, ticker)
    arms = {
        "iid": vpin_sim(seed=seed),
        "regime": vpin_sim(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": vpin_sim(
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
        "kind": "vpin",
        "schema": "vpin.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "claim": "vpin_flow_toxicity_measured_real_vs_sim",
        "interpretation": (
            "VPIN = rolling mean |order imbalance| over equal-volume "
            "buckets. Positive vpin_fwd_vol_corr means toxic flow precedes "
            "volatility — the ELO mechanism. Near-zero real-tape corr is "
            "the honest published result for a single quiet large-cap day."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
