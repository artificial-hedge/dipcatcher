"""touch_follow — what actually moves after a fill on the tape?

``wave23_map.v1`` closed the mechanism hunt with a residual: chased
depth must persist at the touch AND not press it — no composition of
visible placement/cancel knobs expresses that. Before adding another
mechanism, this bench measures what the REAL tape does after each fill:

- **Level-emptying carries instant impact?** Per fill, whether the
  filled touch level emptied (ask steps up after a buy fill), and the
  instant signed drift conditional on it.
- **Which side carries the +4.64-tick continuation?** Exact
  decomposition Δmid = ½(Δbb + Δba) at k=200 into an unhit-side re-site
  (bid steps up after a buy fill) and a hit-side follow-through.
- **Persist without pressing** on the tape = depth accumulating at a
  level while prices migrate: mean unhit-side touch depth k events
  later, old-touch revisit share, and first-passage lag to the unhit
  side stepping.

Same estimator runs on the sim (calibrated base + chase arm) so the gap
is the mechanism's, not the measure's.
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
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

TOUCH_FOLLOW_SCHEMA = "touch_follow.v1"
_K200_LAG = 200


def _kernels(
    bb: np.ndarray,
    ba: np.ndarray,
    dep_bb: np.ndarray,
    dep_ba: np.ndarray,
    fills: np.ndarray,
    signs: np.ndarray,
    *,
    lag: int,
    tick_units: float,
) -> dict[str, Any]:
    """Per-fill forward kernels of touch state (integer price units)."""
    n = len(bb)
    j0 = fills - 1  # pre-fill state index
    valid = fills + lag < n
    J, S, J0 = fills[valid], signs[valid], j0[valid]
    if len(J) == 0:
        return {"n_fills": int(len(fills))}
    mid = 0.5 * (bb + ba)
    # Instant signed drift (per fill, in ticks).
    instant = S * (mid[J] - mid[J0]) / tick_units
    # Touch emptied at fill: buy consumed ask level (ba rises), sell bid.
    emptied = np.where(S > 0, ba[J] > ba[J0], bb[J] < bb[J0])
    # k-window decomposition.
    kJ = J + lag
    d_bb = S * (bb[kJ] - bb[J0]) / (2.0 * tick_units)
    d_ba = S * (ba[kJ] - ba[J0]) / (2.0 * tick_units)
    # Unhit side re-site: buy fill -> bb steps up; first-passage lag.
    resite_lags: list[float] = []
    revisit_shares: list[float] = []
    unhit_dep_growth: list[float] = []
    # Vectorize over the window per fill via loop over fills is too slow;
    # instead aggregate per-fill over a sampled subset for the heavy stats.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(J), size=min(len(J), 4000), replace=False)
    for i in sub:
        j, s = int(J[i]), float(S[i])
        old_bb, old_ba = int(bb[int(J0[i])]), int(ba[int(J0[i])])
        w_bbs = bb[j + 1 : j + 1 + lag]
        w_bas = ba[j + 1 : j + 1 + lag]
        if s > 0:  # buy fill: unhit side is bid; re-site = bb steps up
            hit = np.nonzero(w_bbs > old_bb)[0]
            revisit = w_bas == old_ba
            growth = dep_bb[j + lag] - dep_bb[int(J0[i])]
        else:
            hit = np.nonzero(w_bas < old_ba)[0]
            revisit = w_bbs == old_bb
            growth = dep_ba[j + lag] - dep_ba[int(J0[i])]
        if len(hit):
            resite_lags.append(float(hit[0] + 1))
        revisit_shares.append(float(revisit.mean()))
        unhit_dep_growth.append(float(growth))
    out = {
        "n_fills": int(len(J)),
        "instant_signed_ticks": round(float(instant.mean()), 4),
        "touch_empty_rate": round(float(emptied.mean()), 4),
        "instant_given_empty_ticks": (
            None if not emptied.any() else round(float(instant[emptied].mean()), 4)
        ),
        "instant_given_kept_ticks": (
            None if emptied.all() else round(float(instant[~emptied].mean()), 4)
        ),
        "k200_ticks": round(float((S * (mid[kJ] - mid[J0])).mean()) / tick_units, 4),
        "k200_unhit_share": round(float(d_bb.mean() / (d_bb.mean() + d_ba.mean())), 4)
        if (d_bb.mean() + d_ba.mean()) != 0
        else None,
        "k200_unhit_ticks": round(float(d_bb.mean()), 4),
        "k200_hit_ticks": round(float(d_ba.mean()), 4),
        "unhit_resite_lag_events": (
            None if not resite_lags else round(float(np.mean(resite_lags)), 2)
        ),
        "old_touch_revisit_share": (
            None if not revisit_shares else round(float(np.mean(revisit_shares)), 4)
        ),
        "unhit_touch_depth_growth_k200": (
            None if not unhit_dep_growth else round(float(np.mean(unhit_dep_growth)), 4)
        ),
    }
    return out


def touch_follow_lobster(
    message_path: Path, orderbook_path: Path, *, lag: int = _K200_LAG
) -> dict[str, Any]:
    """Replay the real tape; collect touch state per event + fill signs."""
    book = LobsterBook()
    bb: list[int] = []
    ba: list[int] = []
    dep_bb: list[int] = []
    dep_ba: list[int] = []
    fills: list[int] = []
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
            bb.append(bids[0][0])
            ba.append(asks[0][0])
            dep_bb.append(bids[0][1])
            dep_ba.append(asks[0][1])
            if ev.event_type == EXECUTION:
                fills.append(len(bb) - 1)
                signs.append(-ev.direction)
    return _kernels(
        np.asarray(bb, dtype=np.int64),
        np.asarray(ba, dtype=np.int64),
        np.asarray(dep_bb, dtype=np.int64),
        np.asarray(dep_ba, dtype=np.int64),
        np.asarray(fills, dtype=np.int64),
        np.asarray(signs, dtype=np.int64),
        lag=lag,
        tick_units=100.0,  # LOBSTER prices are price*10_000; tick = 0.01
    )


def touch_follow_sim(
    extra: dict[str, Any] | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
    lag: int = _K200_LAG,
) -> dict[str, Any]:
    """Same estimator on a sim arm (calibrated wave-23 base + overrides)."""
    cfg = _calibrated(seed, extra)
    sim = ZILobSimulator(cfg, _split(3.0, seed + 1))
    bb: list[int] = []
    ba: list[int] = []
    dep_bb: list[int] = []
    dep_ba: list[int] = []
    fills: list[int] = []
    signs: list[int] = []
    seen = 0
    for _ in range(horizon):
        sim.step()
        bb_l, ba_l = sim.best_bid_level, sim.best_ask_level
        if bb_l is None or ba_l is None:
            continue
        bb.append(bb_l)
        ba.append(ba_l)
        dep_bb.append(len(sim._bids[bb_l]))  # noqa: SLF001
        dep_ba.append(len(sim._asks[ba_l]))  # noqa: SLF001
        while seen < len(sim.trades):
            tr = sim.trades[seen]
            fills.append(len(bb) - 1)
            signs.append(1 if tr.aggressor == "buy" else -1)
            seen += 1
    return _kernels(
        np.asarray(bb, dtype=np.int64),
        np.asarray(ba, dtype=np.int64),
        np.asarray(dep_bb, dtype=np.int64),
        np.asarray(dep_ba, dtype=np.int64),
        np.asarray(fills, dtype=np.int64),
        np.asarray(signs, dtype=np.int64),
        lag=lag,
        tick_units=1.0,  # sim levels are already ticks
    )


def touch_follow_bench(
    tape_dir: Path, ticker: str = "AMZN", *, seed: int = 7, horizon: int = 20000
) -> dict[str, Any]:
    """Tape vs sim arms on the touch-follow decomposition."""
    msg = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv"
    ob = tape_dir / f"{ticker}_2012-06-21_34200000_57600000_orderbook_10.csv"
    tape = touch_follow_lobster(msg, ob)
    sim_base = touch_follow_sim(None, horizon=horizon, seed=seed)
    sim_chase = touch_follow_sim(
        {"unhit_imp_frac": 0.5, "unhit_imp_window": 200}, horizon=horizon, seed=seed + 9
    )
    claims = {
        "tape_replay_evaluated": tape.get("n_fills", 0) > 0,
        # Instant impact on the tape is mediated by touch emptying.
        "instant_is_emptying": bool(
            tape.get("instant_given_empty_ticks") is not None
            and tape.get("instant_given_kept_ticks") is not None
            and tape["instant_given_empty_ticks"] > tape["instant_given_kept_ticks"]
        ),
        # Continuation share carried by the unhit side (bid re-siting up
        # after a buy fill) on the tape — the press-without-press channel.
        "tape_continuation_unhit_share": tape.get("k200_unhit_share"),
        # Sim arms carry the same measurement for direct comparison.
        "sim_measured": bool(sim_base.get("n_fills", 0) > 0 and sim_chase.get("n_fills", 0) > 0),
    }
    payload: dict[str, Any] = {
        "schema": TOUCH_FOLLOW_SCHEMA,
        "kind": "sim_bench",
        "data_label": "MIXED",
        "research_only": True,
        "git_revision": git_revision(),
        "seed": seed,
        "horizon_events": horizon,
        "lag": _K200_LAG,
        "tape": tape,
        "sim_base": sim_base,
        "sim_chase": sim_chase,
        "claims": claims,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
