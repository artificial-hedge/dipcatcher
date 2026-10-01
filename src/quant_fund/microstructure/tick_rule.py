"""tick_rule — how wrong is trade-sign inference on labeled tape?

LOBSTER EXECUTION events carry the true aggressor side. Researchers
without direction labels infer it with the tick test (same price →
carry the previous nonzero move's sign) or the quote test (price vs
prev mid). Lee–Ready (1991) report ~15-20% misclassification; knowing
the realized error on THIS tape calibrates every downstream sign-based
statistic (run lengths, VPIN, propagator).

Two classifiers measured against ground truth:
- quote rule: sign = +1 if price > prev mid, -1 if below, else fall
  back to tick rule
- tick rule: sign = sign(last nonzero price change), 0 if none yet
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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

EXEC_TYPES = frozenset({4, 5})  # EXECUTION, EXECUTION_HIDDEN


def _classify(prices: list[float], mids: list[float | None]) -> tuple[list[int], list[int]]:
    """Return (quote_rule_signs, tick_rule_signs) for each trade."""
    quote_signs: list[int] = []
    tick_signs: list[int] = []
    prev_price = prices[0] - 1.0  # force first diff nonzero-ish; guarded below
    last_move_sign = 0
    for i, p in enumerate(prices):
        diff = p - (prices[i - 1] if i else prev_price)
        if diff > 0:
            last_move_sign = 1
        elif diff < 0:
            last_move_sign = -1
        tick_signs.append(last_move_sign)
        mid = mids[i]
        if mid is None or p == mid:
            quote_signs.append(last_move_sign)
        else:
            quote_signs.append(1 if p > mid else -1)
    return quote_signs, tick_signs


def _error_stats(true_signs: list[int], pred: list[int], name: str) -> dict[str, Any]:
    n = len(true_signs)
    undecided = sum(1 for s in pred if s == 0)
    wrong = sum(1 for t, s in zip(true_signs, pred, strict=True) if s != 0 and s != t)
    decided = n - undecided
    return {
        "rule": name,
        "n": n,
        "undecided": undecided,
        "misclass_rate": round(wrong / decided, 4) if decided else None,
    }


def lobster_sign_errors(tape_dir: Path, ticker: str = "AMZN") -> dict[str, Any]:
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    prices: list[float] = []
    mids: list[float | None] = []
    true_signs: list[int] = []
    with ob.open() as fo:
        for ev, ob_row in zip(parse_messages(msg), csv.reader(fo), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            mid = (asks[0][0] + bids[0][0]) / 200.0 if asks and bids else None
            if ev.event_type in EXEC_TYPES:
                prices.append(ev.price / 100.0)  # raw→ticks
                mids.append(mid)
                true_signs.append(-ev.direction)  # resting side → flip
    if not true_signs:
        return {"ok": False, "reason": "no_executions"}
    qs, ts = _classify(prices, mids)
    return {
        "n_execs": len(true_signs),
        "quote_rule": _error_stats(true_signs, qs, "quote_rule"),
        "tick_rule": _error_stats(true_signs, ts, "tick_rule"),
    }


def sim_sign_errors(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> dict[str, Any]:
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    prices: list[float] = []
    mids: list[float | None] = []
    true_signs: list[int] = []
    n_before = 0
    for _ in range(horizon):
        sim.step()
        for tr in sim.trades[n_before:]:
            prices.append(tr.price / cfg.tick)
            mids.append(sim.mid / cfg.tick if sim.mid is not None else None)
            true_signs.append(1 if tr.aggressor == "buy" else -1)
        n_before = len(sim.trades)
    if not true_signs:
        return {"ok": False, "reason": "no_trades"}
    qs, ts = _classify(prices, mids)
    return {
        "n_execs": len(true_signs),
        "quote_rule": _error_stats(true_signs, qs, "quote_rule"),
        "tick_rule": _error_stats(true_signs, ts, "tick_rule"),
    }


def tick_rule_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    """Labeled misclassification rates, real vs sim. Sealed."""
    real = lobster_sign_errors(tape_dir, ticker)
    arms = {
        "iid": sim_sign_errors(seed=seed),
        "regime": sim_sign_errors(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_sign_errors(
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
        "kind": "tick_rule",
        "schema": "tick_rule.v1",
        "ticker": ticker,
        "real": real,
        "sim_arms": arms,
        "claim": "sign_inference_misclassification_measured",
        "interpretation": (
            "Real LOBSTER tape has ground-truth aggressor labels; the "
            "misclass rates measure how much noise sign inference would "
            "inject. On sim, fills at the touch mostly classify right — "
            "hidden/mid fills are where rules fail."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
