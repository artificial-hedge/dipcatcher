"""imbalance_predict — does touch imbalance predict the NEXT event's sign?

For every signed event on the tape, condition on the touch imbalance
``i = (bid_sz - ask_sz) / (bid_sz + ask_sz)`` at the *prior* book state
(the orderbook row before the event — LOBSTER rows are post-event
snapshots, so row k-1 is the state event k reacts to) and record the
event's signed aggressor indicator:

- EXECUTION / EXECUTION_HIDDEN → aggressor side = -direction
- SUBMISSION → direction (a resting buy add is bullish)
- CANCEL_PARTIAL / DELETE → -direction (a bid cancel is bearish)

Measurement: P(next signed event = buy) and mean signed flow per
imbalance quintile, plus the same conditioned on event type (exec vs
submission). Quintile edges come from the pooled imbalance sample so
the conditioned tables share bins. The headline is the OLS slope of
P(buy) across quintile indices — a monotone imbalance→direction
gradient is the queue-reactive predictability the ZI-LOB sim should
only partially reproduce.

Sim arm: the ZI-LOB book depth IS exposed (``depth_at`` + best levels),
so the same imbalance conditioning applies; per-event sides are read
from the returned event type plus book-depth deltas (the sim's event
stream has no placement choice, but the mechanism is present).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    SUBMISSION,
    LobsterEvent,
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

IMBALANCE_PREDICT_SCHEMA = "imbalance_predict.v1"

MIN_SAMPLES = 50
DIVERGENCE_SLOPE = 0.5

Sample = tuple[float, int, str]  # (prior touch imbalance, sign, kind)


def _lobster_sign(ev: LobsterEvent) -> tuple[str, int] | None:
    """(kind, ±1) for a signed event; None when the event carries no sign."""
    if ev.event_type in (EXECUTION, EXECUTION_HIDDEN):
        return "exec", -ev.direction  # direction = resting side
    if ev.event_type == SUBMISSION:
        return "submission", ev.direction
    if ev.event_type in (CANCEL_PARTIAL, DELETE):
        return "cancel", -ev.direction
    return None


def _round(x: float, nd: int = 4) -> float | None:
    return round(float(x), nd) if math.isfinite(float(x)) else None


def _ols_slope(ys: list[float | None]) -> float | None:
    """OLS slope over indices of populated quintiles (None = empty bin)."""
    pts = [(i, y) for i, y in enumerate(ys) if y is not None]
    if len(pts) < 2:
        return None
    x = np.asarray([p[0] for p in pts], dtype=float)
    y = np.asarray([p[1] for p in pts], dtype=float)
    return _round(float(np.polyfit(x, y, 1)[0]))


def _binned(imb: np.ndarray, sign: np.ndarray, edges: np.ndarray) -> dict[str, Any]:
    """Per-quintile n / P(buy) / mean signed flow at pooled edges."""
    idx = np.searchsorted(edges, imb, side="right")  # 0..4
    rows: list[dict[str, Any]] = []
    for q in range(5):
        mask = idx == q
        n = int(mask.sum())
        if n == 0:
            rows.append({"quintile": q, "n": 0})
            continue
        rows.append(
            {
                "quintile": q,
                "n": n,
                "imb_lo": _round(float(imb[mask].min())),
                "imb_hi": _round(float(imb[mask].max())),
                "p_buy": _round(float(np.mean(sign[mask] == 1))),
                "mean_signed_flow": _round(float(sign[mask].mean())),
            }
        )
    return {
        "quintiles": rows,
        "p_buy_slope_per_quintile": _ols_slope([r.get("p_buy") for r in rows]),
        "mean_signed_flow_slope_per_quintile": _ols_slope(
            [r.get("mean_signed_flow") for r in rows]
        ),
    }


def _measure(samples: list[Sample]) -> dict[str, Any]:
    """Quintile tables over (imbalance, sign, kind) samples."""
    n = len(samples)
    if n < MIN_SAMPLES:
        return {"ok": False, "n": n, "reason": "too_few_samples"}
    imb = np.asarray([s[0] for s in samples], dtype=float)
    sign = np.asarray([s[1] for s in samples], dtype=int)
    kinds = [s[2] for s in samples]
    edges = np.asarray(np.quantile(imb, [0.2, 0.4, 0.6, 0.8]), dtype=float)
    by_type: dict[str, Any] = {}
    for kind in ("exec", "submission"):
        mask = np.asarray([k == kind for k in kinds])
        table = _binned(imb[mask], sign[mask], edges)
        table["n"] = int(mask.sum())
        by_type[kind] = table
    return {
        "ok": True,
        "n": n,
        "n_by_type": {k: kinds.count(k) for k in ("exec", "submission", "cancel")},
        "overall_p_buy": _round(float(np.mean(sign == 1))),
        "overall_mean_signed_flow": _round(float(sign.mean())),
        "quintile_edges": [_round(float(e)) for e in edges],
        "by_type": by_type,
        **_binned(imb, sign, edges),
    }


def _lobster_samples(msg_path: Path, ob_path: Path) -> list[Sample]:
    """(prior-row touch imbalance, event sign, kind) per signed event."""
    samples: list[Sample] = []
    prev_imb: float | None = None
    with ob_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            signed = _lobster_sign(ev)
            if prev_imb is not None and signed is not None:
                kind, sign = signed
                samples.append((prev_imb, sign, kind))
            asks, bids = parse_orderbook_row(ob_row)
            if asks and bids:
                a_sz, b_sz = float(asks[0][1]), float(bids[0][1])
                prev_imb = (b_sz - a_sz) / (a_sz + b_sz) if (a_sz + b_sz) > 0 else None
            else:
                prev_imb = None
    return samples


def _sim_samples(sim: ZILobSimulator, horizon: int) -> list[Sample]:
    """Same pairing on the sim tape: pre-step touch, post-step event sign.

    step() returns the event type; the sign comes from the trades delta
    (market) or the book-depth delta (limit rests +1 on its side, cancel
    removes -1 from its side). Events with no observable sign — a market
    order that hits an empty book, a cancel with no resting depth — are
    skipped, as are events whose prior touch does not exist.
    """
    samples: list[Sample] = []
    for _ in range(horizon):
        ba_lvl, bb_lvl = sim.best_ask_level, sim.best_bid_level
        prev_imb: float | None = None
        if ba_lvl is not None and bb_lvl is not None:
            a_sz = float(sim.depth_at("sell", ba_lvl))
            b_sz = float(sim.depth_at("buy", bb_lvl))
            if a_sz + b_sz > 0:
                prev_imb = (b_sz - a_sz) / (a_sz + b_sz)
        b_d, a_d = sim.bid_depth, sim.ask_depth
        n_tr = len(sim.trades)
        etype = sim.step()
        sign: int | None = None
        kind = "exec" if etype == "market" else ("submission" if etype == "limit" else "cancel")
        if etype == "market":
            if len(sim.trades) > n_tr:
                sign = 1 if sim.trades[-1].aggressor == "buy" else -1
        elif etype == "limit":
            if sim.bid_depth > b_d:
                sign = 1
            elif sim.ask_depth > a_d:
                sign = -1
        elif sim.bid_depth < b_d:
            sign = -1
        elif sim.ask_depth < a_d:
            sign = 1
        if prev_imb is not None and sign is not None:
            samples.append((prev_imb, sign, kind))
    return samples


def lobster_imbalance_predict(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    return _measure(_lobster_samples(msg_path, ob_path))


def sim_imbalance_predict(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    *,
    horizon: int = 30_000,
    seed: int = 7,
) -> dict[str, Any]:
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    out = _measure(_sim_samples(sim, horizon))
    out["mechanism_present"] = True  # book depth exposed via depth_at
    return out


def imbalance_predict_bench(
    tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7
) -> dict[str, Any]:
    """Imbalance→next-sign predictability, real tape vs sim arms. Sealed."""
    msg_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob_path = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    if not msg_path.exists() or not ob_path.exists():
        raise FileNotFoundError(f"LOBSTER tape not found in {tape_dir}")
    real = lobster_imbalance_predict(msg_path, ob_path)
    arms = {
        "iid": sim_imbalance_predict(seed=seed),
        "regime": sim_imbalance_predict(
            MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_imbalance_predict(
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
    rs = real.get("p_buy_slope_per_quintile")
    if rs is not None:
        for name, arm in arms.items():
            s = arm.get("p_buy_slope_per_quintile")
            if s is None:
                continue
            if rs * s < 0 or abs(rs - s) > DIVERGENCE_SLOPE:
                divergences.append(f"{name}_slope_{s:.3f}_vs_{rs:.3f}")
    payload: dict[str, Any] = {
        "kind": "imbalance_predict",
        "schema": IMBALANCE_PREDICT_SCHEMA,
        "ticker": ticker,
        "tape_sha256": hash_bytes(msg_path.read_bytes()),
        "tape_files": [msg_path.name, ob_path.name],
        "real": real,
        "sim_arms": arms,
        "headline": {"p_buy_slope_per_quintile": rs},
        "divergences": divergences,
        "claim": "touch_imbalance_predicts_next_sign_measured",
        "interpretation": (
            "Each signed event is scored against the touch imbalance at "
            "the prior book row. p_buy_slope_per_quintile is the OLS "
            "slope of P(next event = buy) across imbalance quintiles; a "
            "positive monotone gradient means queue skew anticipates the "
            "next event's direction. Quintile edges are pooled so the "
            "exec/submission-conditioned tables share bins. ZI-LOB arms "
            "re-use the same machinery on the sim's own book — its "
            "symmetric flows leave slope near zero unless a flow driver "
            "(regime/split) injects direction persistence."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
