"""Queue-priority lane: fill probability as a function of queue position.

Measures, on the **SYNTHETIC** zero-intelligence LOB
(``microstructure.zi_lob_simulator``; Cont-Stoikov-Talreja 2010 queueing,
Santa Fe / Moret & Lillo 2026 calibration), how a resting unit order's
position in its price-level FIFO queue at submit time predicts whether and
how fast it is filled. Two arms are compared: iid zero-intelligence flow and
a two-state ``MarkovRegimeFlow`` modulation on the market-order clock.

Measurements:

- ``fill_curve(trades)`` — the fill-side view: the histogram (and pmf) of
  ``maker_queue_ahead_at_submit`` across executed trades, plus a fill-time
  proxy ``t - maker_t_submit`` per trade and per queue position. A
  distribution heavy at 0 means price priority regenerates fresh queue heads
  faster than time priority binds.
- ``fill_probability_by_queue(submissions, trades)`` — the submission-side
  view: joins the fill distribution to the **resting-order submission
  stream**, so the denominator is every order rested, not just winners.
  ``P(eventual fill | queue_ahead at submit)`` needs orders that were
  submitted and never filled; the ZI simulator exposes them through the
  public registry — order ids are sequential
  (``event_counts()["n_orders_created"]`` is the ``_rest`` counter) and each
  live id answers ``queue_ahead_at_submit`` / ``order_alive``. Because the
  engine applies at most one mutation per ``step()``, an order created during
  a step is necessarily alive right after it: polling new ids each step
  recovers the complete submission stream without touching private state.
  Orders resting at measurement end are right-censored (eventual outcome
  unknown), not treated as unfilled.

Honesty: every output is a labeled SYNTHETIC correctness diagnostic, never
market evidence; no live-trading claim, no PnL headlining. Fail-closed:
empty trade/submission streams raise instead of fabricating rates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    RegimeState,
    TradeEvent,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

QUEUE_PRIORITY_SCHEMA = "queue_priority.v1"
QUEUE_PRIORITY_KIND = "queue_priority"
QUEUE_PRIORITY_RECEIPT_PATH = Path("receipts/queue_priority.json")

Outcome = Literal["filled", "canceled", "resting"]


@dataclass(frozen=True)
class QueueSubmission:
    """One resting order resolved against the measurement window.

    ``resting=True`` means the order was still live when measurement ended:
    right-censored, eventual fill unknown — never counted as unfilled.
    """

    order_id: int
    queue_ahead_at_submit: int
    resting: bool


def _as_counter(trades: Sequence[TradeEvent]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for tr in trades:
        hist[tr.maker_queue_ahead_at_submit] = hist.get(tr.maker_queue_ahead_at_submit, 0) + 1
    return hist


def _fill_waits(trades: Sequence[TradeEvent]) -> dict[int, list[float]]:
    by_q: dict[int, list[float]] = {}
    for tr in trades:
        by_q.setdefault(tr.maker_queue_ahead_at_submit, []).append(tr.t - tr.maker_t_submit)
    return by_q


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fill_curve(trades: Sequence[TradeEvent]) -> dict[str, Any]:
    """Fill-side queue-position distribution over executed trades.

    Histogram (and pmf) of ``maker_queue_ahead_at_submit`` — how many unit
    orders were ahead of each *filled* maker when it rested — plus the
    fill-time proxy ``t - maker_t_submit`` overall and per queue position.
    ``front_dominant`` is True when queue position 0 is the modal outcome:
    price priority refilling a fresh head beats time priority.

    Fail-closed: an empty trade stream raises ``ValueError``.
    """
    if not trades:
        raise ValueError("fill_curve requires a non-empty trade stream")
    hist = _as_counter(trades)
    n = len(trades)
    pmf = {str(q): count / n for q, count in sorted(hist.items())}
    waits = _fill_waits(trades)
    all_waits = [tr.t - tr.maker_t_submit for tr in trades]
    per_q = {
        str(q): {
            "n_fills": len(ws),
            "mean_wait_seconds": float(np.mean(ws)),
            "median_wait_seconds": _percentile(ws, 50.0),
        }
        for q, ws in sorted(waits.items())
    }
    ahead = np.asarray([tr.maker_queue_ahead_at_submit for tr in trades], dtype=np.float64)
    mode_q = max(hist.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return {
        "label": "SYNTHETIC",
        "n_fills": n,
        "queue_ahead_hist": {str(q): c for q, c in sorted(hist.items())},
        "queue_ahead_pmf": pmf,
        "p_queue_ahead_zero": float(hist.get(0, 0) / n),
        "mode_queue_ahead": int(mode_q),
        "front_dominant": mode_q == 0,
        "mean_queue_ahead": float(ahead.mean()),
        "median_queue_ahead": _percentile(ahead.tolist(), 50.0),
        "p90_queue_ahead": _percentile(ahead.tolist(), 90.0),
        "max_queue_ahead": int(ahead.max()),
        "mean_fill_wait_seconds": float(np.mean(all_waits)),
        "median_fill_wait_seconds": _percentile(all_waits, 50.0),
        "p90_fill_wait_seconds": _percentile(all_waits, 90.0),
        "max_fill_wait_seconds": float(np.max(all_waits)),
        "fill_wait_by_queue": per_q,
    }


def fill_probability_by_queue(
    submissions: Sequence[QueueSubmission],
    trades: Sequence[TradeEvent],
) -> dict[str, Any]:
    """P(eventual fill | queue_ahead at submit) over the submission stream.

    Joins the fill-side curve to every order that rested — the denominator
    includes orders canceled or never reached, which the trade stream alone
    cannot see. Per queue position ``q``: submissions, fills, cancellations,
    right-censored still-resting orders, and ``p_fill = n_filled /
    n_submitted`` (a lower bound on eventual fill probability whenever
    censored orders remain). ``p_cancel`` counts window-resolved
    cancellations; ``p_censored`` marks the unresolved share.

    Fail-closed: an empty submission stream raises ``ValueError``.
    """
    if not submissions:
        raise ValueError("fill_probability_by_queue requires a non-empty submission stream")
    filled_ids = {tr.maker_order_id for tr in trades}
    wait_of = {tr.maker_order_id: tr.t - tr.maker_t_submit for tr in trades}
    cells: dict[int, dict[str, Any]] = {}
    n_filled = n_canceled = n_resting = 0
    for sub in submissions:
        if sub.order_id in filled_ids:
            outcome: Outcome = "filled"
            n_filled += 1
        elif sub.resting:
            outcome = "resting"
            n_resting += 1
        else:
            outcome = "canceled"
            n_canceled += 1
        cell = cells.setdefault(
            sub.queue_ahead_at_submit,
            {"n_submitted": 0, "n_filled": 0, "n_canceled": 0, "n_resting": 0, "_waits": []},
        )
        cell["n_submitted"] += 1
        cell[f"n_{outcome}"] += 1
        if outcome == "filled":
            waits = cell["_waits"]
            if not (isinstance(waits, list)):
                raise ValueError("isinstance(waits, list)")
            waits.append(wait_of[sub.order_id])
    n = len(submissions)
    per_q: dict[str, Any] = {}
    for q, cell in sorted(cells.items()):
        ns = int(cell["n_submitted"])
        waits = cell.pop("_waits")
        per_q[str(q)] = {
            **cell,
            "p_fill": cell["n_filled"] / ns,
            "p_cancel": cell["n_canceled"] / ns,
            "p_censored": cell["n_resting"] / ns,
            "mean_fill_wait_seconds": float(np.mean(waits)) if waits else None,
        }
    return {
        "label": "SYNTHETIC",
        "n_submissions": n,
        "n_filled": n_filled,
        "n_canceled": n_canceled,
        "n_resting_censored": n_resting,
        "p_fill_overall": n_filled / n,
        "censoring_fraction": n_resting / n,
        "per_queue_position": per_q,
    }


def _track_window(
    sim: ZILobSimulator, horizon: float
) -> tuple[list[QueueSubmission], int, list[TradeEvent]]:
    """Step ``sim`` for ``horizon`` seconds, tracking every new resting order.

    Order ids are the sequential ``_rest`` counter surfaced as
    ``n_orders_created``; each new id is polled for ``queue_ahead_at_submit``
    immediately after the step that created it. The engine applies at most
    one mutation per step, so a just-created order is necessarily alive —
    the full submission stream (including orders later canceled) is captured
    without private state. Returns (submissions, n_untracked) where
    n_untracked counts ids that exited before their queue state was
    readable — only possible for orders created before tracking started.
    """
    horizon = float(horizon)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError(f"horizon must be positive and finite, got {horizon!r}")
    t_end = sim.t + horizon
    seen = sim.event_counts()["n_orders_created"]
    ahead: dict[int, int] = {}
    n_untracked = 0
    for oid in range(seen):
        qa = sim.queue_ahead_at_submit(oid)
        if qa is None:
            n_untracked += 1  # exited before the measurement window opened
        else:
            ahead[oid] = qa
    n_trades0 = len(sim.trades)
    while sim.t < t_end:
        sim.step()
        n_created = sim.event_counts()["n_orders_created"]
        for oid in range(seen, n_created):
            qa = sim.queue_ahead_at_submit(oid)
            if qa is None:
                n_untracked += 1
            else:
                ahead[oid] = qa
        seen = n_created
    trades = sim.trades[n_trades0:]
    subs = [
        QueueSubmission(order_id=oid, queue_ahead_at_submit=q, resting=sim.order_alive(oid))
        for oid, q in sorted(ahead.items())
    ]
    return subs, n_untracked, trades


def run_queue_arm(
    config: ZILobConfig,
    *,
    horizon: float,
    warmup: float = 0.0,
    flow: MarkovRegimeFlow | None = None,
    flow_label: str = "iid",
) -> dict[str, Any]:
    """Run one ZI-LOB arm and return the queue-priority measurement bundle.

    ``warmup`` seconds of equilibration are stepped first; the measurement
    window then tracks submissions for ``horizon`` seconds. Orders resting at
    window open stay in the denominator (in-flight submissions whose outcomes
    resolve inside the window); orders that already exited are reported as
    ``n_untracked_pre_window``. Fail-closed via ``fill_curve`` when the window
    produces no fills.
    """
    wu = float(warmup)
    if not math.isfinite(wu) or wu < 0.0:
        raise ValueError(f"warmup must be non-negative and finite, got {warmup!r}")
    sim = ZILobSimulator(config, flow=flow)
    if wu > 0.0:
        sim.run(sim.t + wu)
    subs, n_untracked, trades = _track_window(sim, horizon)
    return {
        "label": "SYNTHETIC",
        "flow": flow_label,
        "seed": config.seed,
        "horizon_seconds": float(horizon),
        "warmup_seconds": wu,
        "n_tracked_submissions": len(subs),
        "n_untracked_pre_window": n_untracked,
        "event_counts": sim.event_counts(),
        "fill_curve": fill_curve(trades),
        "queue_survival": fill_probability_by_queue(subs, trades),
    }


def queue_priority_bench(
    *,
    seed: int = 7,
    horizon: float = 2000.0,
    warmup: float = 300.0,
    out_path: Path | str = QUEUE_PRIORITY_RECEIPT_PATH,
) -> dict[str, Any]:
    """Queue-priority receipt over the iid + MarkovRegimeFlow arms.

    Seals ``payload["receipt_sha256"]`` over the canonical payload and writes
    ``receipts/queue_priority.json`` (``indent=2``, ``sort_keys``,
    ``allow_nan=False`` — no NaN/Infinity anywhere in the payload). Returns
    the sealed payload.
    """
    iid = run_queue_arm(ZILobConfig(seed=seed), horizon=horizon, warmup=warmup, flow_label="iid")
    flow = MarkovRegimeFlow(
        states=(
            RegimeState("calm", 1.0, 0.5),
            RegimeState("bursty", 3.0, 0.62),
        ),
        stay_probs=(0.995, 0.985),
        seed=seed + 1,
    )
    markov = run_queue_arm(
        ZILobConfig(seed=seed + 2),
        horizon=horizon,
        warmup=warmup,
        flow=flow,
        flow_label="markov_regime",
    )
    front_iid = iid["fill_curve"]["p_queue_ahead_zero"]
    front_mk = markov["fill_curve"]["p_queue_ahead_zero"]
    interpretation = [
        (
            f"iid arm: {front_iid:.1%} of fills rested at queue position 0 at submit; "
            f"markov arm: {front_mk:.1%}. A histogram heavy at 0 means price-time "
            "priority resolves at fresh queue heads more often than back-of-queue "
            "orders survive to the touch."
        ),
        (
            "queue_survival joins the fill-side curve to the full resting-order "
            "submission stream (public registry: sequential order ids + "
            "queue_ahead_at_submit/order_alive), so P(fill | queue position at "
            "submit) includes submitted-but-never-filled orders; orders still "
            "resting at measurement end are reported as right-censored, not "
            "unfilled."
        ),
        (
            "Fill-wait seconds (t - maker_t_submit) is the survival/fill-time "
            "proxy; per-position cancellation shares are window-resolved. All "
            "results are SYNTHETIC simulator diagnostics, not market evidence."
        ),
    ]
    payload: dict[str, Any] = {
        "kind": QUEUE_PRIORITY_KIND,
        "schema": QUEUE_PRIORITY_SCHEMA,
        "level": "research",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "synthetic_simulator_diagnostic_only",
        "data_source": ZI_LOB_REVISION,
        "code_revision": git_revision(),
        "params": {
            "seed": seed,
            "horizon_seconds": float(horizon),
            "warmup_seconds": float(warmup),
            "sim": "ZILobSimulator",
            "arms": ["iid", "markov_regime"],
            "markov_flow": {
                "states": [
                    {"name": "calm", "intensity_mult": 1.0, "p_buy": 0.5},
                    {"name": "bursty", "intensity_mult": 3.0, "p_buy": 0.62},
                ],
                "stay_probs": [0.995, 0.985],
            },
        },
        "arms": {"iid": iid, "markov_regime": markov},
        "comparison": {
            "p_queue_ahead_zero_iid": front_iid,
            "p_queue_ahead_zero_markov": front_mk,
            "p_fill_overall_iid": iid["queue_survival"]["p_fill_overall"],
            "p_fill_overall_markov": markov["queue_survival"]["p_fill_overall"],
            "mean_fill_wait_seconds_iid": iid["fill_curve"]["mean_fill_wait_seconds"],
            "mean_fill_wait_seconds_markov": markov["fill_curve"]["mean_fill_wait_seconds"],
        },
        "interpretation": interpretation,
        "evidence": [
            "zi_lob_seeded_run",
            "fill_side_queue_ahead_histogram",
            "submission_stream_fill_probability",
            "fill_wait_survival_proxy",
        ],
        "claims": [
            {
                "text": (
                    "fill probability and fill time measured as a function of "
                    "queue position at submit on the seeded SYNTHETIC ZI-LOB "
                    "(iid + MarkovRegimeFlow arms); submission-side join via the "
                    "public order registry with right-censoring accounting"
                ),
                "kind": "empirical_synthetic",
            }
        ],
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    atomic_write_text(
        Path(out_path),
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return payload


__all__ = [
    "QUEUE_PRIORITY_KIND",
    "QUEUE_PRIORITY_RECEIPT_PATH",
    "QUEUE_PRIORITY_SCHEMA",
    "QueueSubmission",
    "fill_curve",
    "fill_probability_by_queue",
    "queue_priority_bench",
    "run_queue_arm",
]
