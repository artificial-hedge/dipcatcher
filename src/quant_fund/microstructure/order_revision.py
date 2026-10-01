"""order_revision — the re-quote half of the book-breathing cycle.

`cancel_cluster` (#501) measured the retreat: executions trigger a
5-7x cancel burst within 0.5s. This lane measures what happens NEXT:
do the canceled orders re-appear at a near price (a quote revision —
the same liquidity re-priced) or is the liquidity gone?

LOBSTER does not link order ids across a cancel→resubmit (each
resubmission is a fresh id), so the link is a stated heuristic: for
each DELETE/CANCEL event at (side, price, t), find the nearest
same-side SUBMISSION in (t, t+w]; nearest in |Δprice| on ties.
Statistics:

- revision_rate: share of deletes with a linked resubmission
- revision latency: median t_resubmit − t_delete
- revision step: distribution of (resubmit − deleted) price in ticks
  — a tight ±1 tick cluster is re-quoting; a wide shift is a new
  decision
- away-vs-toward: does the revision step toward or away from mid

The sim has no re-quote mechanism (every agent's event is
independent), so the divergence is the mechanism absence itself.
Heuristic, labeled as such in the receipt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    SUBMISSION,
    LobsterEvent,
    parse_messages,
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

WINDOW_S = 0.5  # match cancel_cluster's post-exec burst window


def _link_revisions(
    cancels: list[LobsterEvent],
    submits: list[LobsterEvent],
    window_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each cancel, nearest same-side submit within the window.

    Returns (latencies_s, price_steps, directions of the linked cancels).
    submits must be sorted by time.
    """
    times = np.array([s.time_s for s in submits])
    lats: list[float] = []
    steps: list[float] = []
    dirs: list[float] = []
    for ev in cancels:
        lo = np.searchsorted(times, ev.time_s, side="right")
        hi = np.searchsorted(times, ev.time_s + window_s, side="right")
        cands = [s for s in submits[lo:hi] if s.direction == ev.direction]
        if not cands:
            continue
        best = min(cands, key=lambda s: abs(s.price - ev.price))
        lats.append(best.time_s - ev.time_s)
        steps.append(float(best.price - ev.price))
        dirs.append(float(ev.direction))
    return np.asarray(lats), np.asarray(steps), np.asarray(dirs)


def lobster_revision_stats(msg_path: Path, window_s: float = WINDOW_S) -> dict[str, Any]:
    events = list(parse_messages(msg_path))
    cancels = [e for e in events if e.event_type in (CANCEL_PARTIAL, DELETE)]
    submits = sorted((e for e in events if e.event_type == SUBMISSION), key=lambda e: e.time_s)
    if len(cancels) < 50:
        return {"ok": False, "reason": "too_few_cancels", "n_cancels": len(cancels)}
    lats, steps, dirs = _link_revisions(cancels, submits, window_s)
    n = dirs.size
    tick = 100.0  # price units: dollars*1e4, tick=0.01 → 100 units
    step_ticks = steps / tick
    # toward mid: a buy re-quoting higher steps toward the ask; a sell
    # re-quoting lower steps toward the bid
    toward = int(np.sum((dirs > 0) & (steps > 0)) + np.sum((dirs < 0) & (steps < 0)))
    away = int(np.sum((dirs > 0) & (steps < 0)) + np.sum((dirs < 0) & (steps > 0)))
    return {
        "ok": True,
        "n_cancels": len(cancels),
        "n_submits": len(submits),
        "revision_rate": float(n / len(cancels)),
        "median_latency_ms": float(np.median(lats) * 1000) if n else None,
        "step_ticks_abs_median": float(np.median(np.abs(step_ticks))) if n else None,
        "step_ticks_p90": float(np.percentile(np.abs(step_ticks), 90)) if n else None,
        "same_price_share": float(np.mean(step_ticks == 0)) if n else None,
        "within_3_ticks_share": float(np.mean(np.abs(step_ticks) <= 3)) if n else None,
        "toward_mid_share": float(toward / n) if n else None,
        "away_mid_share": float(away / n) if n else None,
    }


def sim_revision_stats(
    flow: MOFlow | MarkovRegimeFlow | SplitFlow | None = None,
    horizon: int = 30_000,
    window_s: float = WINDOW_S,
    seed: int = 7,
) -> dict[str, Any]:
    """The sim has no revisions; measure the accidental link rate as
    the honest null — the sim's 'revision rate' is just how often a
    random same-side insert lands near a random delete."""
    sim = ZILobSimulator(ZILobConfig(seed=seed), flow=flow)
    cancels: list[LobsterEvent] = []
    submits: list[LobsterEvent] = []
    known: dict[int, tuple[int, int]] = {}  # oid -> (side, level)
    for oid, o in sim._orders.items():  # seed book
        known[oid] = (1 if o.side == "buy" else -1, o.level)
    for _ in range(horizon):
        prev = set(known)
        kind = sim.step()
        cur = {oid: (1 if o.side == "buy" else -1, o.level) for oid, o in sim._orders.items()}
        if kind == "limit":
            for oid in set(cur) - prev:
                sd, lv = cur[oid]
                submits.append(LobsterEvent(sim.t, SUBMISSION, oid, 1, lv, sd))
        elif kind == "cancel":
            for oid in prev - set(cur):
                sd, lv = known[oid]
                cancels.append(LobsterEvent(sim.t, DELETE, oid, 1, lv, sd))
        known = cur
    if len(cancels) < 50:
        return {"ok": False, "reason": "too_few_cancels", "n_cancels": len(cancels)}
    lats, steps, dirs = _link_revisions(cancels, submits, window_s)
    n = dirs.size
    step_ticks = steps  # sim prices already in ticks
    return {
        "ok": True,
        "n_cancels": len(cancels),
        "n_submits": len(submits),
        "revision_rate": float(n / len(cancels)),
        "median_latency_ms": float(np.median(lats) * 1000) if n else None,
        "step_ticks_abs_median": float(np.median(np.abs(step_ticks))) if n else None,
        "within_3_ticks_share": float(np.mean(np.abs(step_ticks) <= 3)) if n else None,
    }


def revision_bench(tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7) -> dict[str, Any]:
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    if not msg:
        raise FileNotFoundError(f"no LOBSTER message CSV under {tape_dir}")
    real = lobster_revision_stats(msg[0])
    arms = {
        "iid": sim_revision_stats(seed=seed),
        "regime": sim_revision_stats(
            flow=MarkovRegimeFlow(
                states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
                stay_probs=(0.995, 0.985),
                seed=seed + 1,
            ),
            seed=seed + 1,
        ),
        "split": sim_revision_stats(
            flow=SplitFlow(
                p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed + 2
            ),
            seed=seed + 2,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        rr = real["revision_rate"]
        for name, arm in arms.items():
            if arm.get("ok") and abs(arm["revision_rate"] - rr) > 0.1:
                divergences.append(f"{name}_rate_{arm['revision_rate']:.3f}_vs_{rr:.3f}")
    payload: dict[str, Any] = {
        "kind": "order_revision",
        "schema": "order_revision.v1",
        "ticker": ticker,
        "window_s": WINDOW_S,
        "heuristic": "nearest same-side submit within window; nearest |Δprice| on ties",
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claim": "requote_cycle_measured_real_vs_sim",
        "interpretation": (
            "revision_rate = share of deletes followed by a linked "
            "resubmission; sim arms carry no revision mechanism so their "
            "rate is the accidental-null. step_ticks within ±3 = "
            "re-quote; toward_mid_share measures whether revisions chase "
            "or retreat from the touch."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
