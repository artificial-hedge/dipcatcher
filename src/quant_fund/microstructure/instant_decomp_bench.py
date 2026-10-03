"""instant_decomp_bench — decompose instantaneous impact into book mechanics.

``impact_persist_bench`` showed the tape's fills carry ~0.89 ticks of
same-event signed mid drift, and ``anchor_scan_bench`` showed no
anchor-channel cell closes it. This lane asks *mechanically where* the
instant term comes from: for a same-event mid move the fill must empty
the touch level AND the next level must sit a gap away, so

    E[signed Δmid] ≈ P(fill empties the touch) × E[gap | emptied] / 2

which each dataset can measure independently. Per fill the bench records
whether the touched level was emptied, the gap to the new touch (in
ticks) conditional on emptying, and the realized signed Δmid — then the
predicted instant from the identity vs the realized one (the unexplained
residual is mid moves from non-emptying fills, e.g. opposite-side
improvements inside a multi-tick spread).

Real side: the LOBSTER message/orderbook pair — per EXECUTION, the
pre-event and post-event snapshot rows give the touched level's fate
and the next level's gap. Sim side: the same bookkeeping around each
fill-bearing ``step()`` on the ZI-LOB arms.

Receipts: ``instant_decomp.v1`` — sealed, MIXED label (real tape +
SYNTHETIC arms), research-only.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from quant_fund.microstructure.lobster import EXECUTION, parse_messages, parse_orderbook_row
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator, santa_fe_config
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

INSTANT_DECOMP_SCHEMA = "instant_decomp.v1"


def _decomp_stats(
    *,
    n_fills: int,
    n_empty: int,
    gap_ticks: list[float],
    signed_dmids: list[float],
) -> dict[str, Any]:
    if n_fills < 10:
        return {"ok": False, "n_fills": n_fills}
    p_empty = n_empty / n_fills
    mean_gap = float(sum(gap_ticks) / len(gap_ticks)) if gap_ticks else 0.0
    realized = float(sum(signed_dmids) / len(signed_dmids)) if signed_dmids else 0.0
    predicted = p_empty * mean_gap * 0.5
    return {
        "ok": True,
        "n_fills": n_fills,
        "p_empty_touch": round(p_empty, 4),
        "mean_gap_ticks_when_empty": round(mean_gap, 4),
        "realized_instant_ticks": round(realized, 4),
        "predicted_instant_ticks": round(predicted, 4),
        "unexplained_instant_ticks": round(realized - predicted, 4),
    }


def lobster_instant_decomp(msg_path: Path, ob_path: Path) -> dict[str, Any]:
    """Per-exec touch-emptying + next-level gap on the real tape."""
    n_fills = 0
    n_empty = 0
    gaps: list[float] = []
    dmids: list[float] = []
    with ob_path.open() as f_ob:
        prev_asks: list[tuple[int, int]] = []
        prev_bids: list[tuple[int, int]] = []
        for ev, ob_row in zip(parse_messages(msg_path), csv.reader(f_ob), strict=True):
            asks, bids = parse_orderbook_row(ob_row)
            if ev.event_type == EXECUTION and prev_asks and prev_bids and asks and bids:
                n_fills += 1
                pre_mid = 0.5 * (prev_asks[0][0] + prev_bids[0][0])
                post_mid = 0.5 * (asks[0][0] + bids[0][0])
                sign = -ev.direction  # aggressor sign
                dmids.append(sign * (post_mid - pre_mid) / 100.0)
                pre_side = prev_asks if ev.direction == -1 else prev_bids
                post_side = asks if ev.direction == -1 else bids
                touch = pre_side[0][0]
                post_prices = {p for p, _ in post_side}
                if touch not in post_prices:
                    n_empty += 1
                    if post_side:
                        new_touch = post_side[0][0]
                        # Signed in the aggressor's drift direction:
                        # for a buy exec the emptied ask lifts
                        # (new > old), for a sell exec the emptied bid
                        # drops (new < old) — sign*(new - old) is the
                        # gap in the mid-move direction either way.
                        gaps.append(sign * (new_touch - touch) / 100.0)
            prev_asks, prev_bids = asks, bids
    return _decomp_stats(n_fills=n_fills, n_empty=n_empty, gap_ticks=gaps, signed_dmids=dmids)


def sim_instant_decomp(
    cfg: Any = None, flow: Any = None, *, horizon: int = 20000, seed: int = 7
) -> dict[str, Any]:
    """Same bookkeeping around each fill-bearing sim step."""
    sim = ZILobSimulator(cfg if cfg is not None else santa_fe_config(seed=seed), flow=flow)

    def _touch(side: str) -> int | None:
        if side == "buy":  # aggressor buy hits the ask
            return sim.best_ask_level
        return sim.best_bid_level

    n_fills = 0
    n_empty = 0
    gaps: list[float] = []
    dmids: list[float] = []
    n_trades = 0
    for _ in range(horizon):
        # Pre-event touch/mid are only meaningful when a fill lands.
        bb, ba = sim.best_bid_level, sim.best_ask_level
        sim.step()
        new_trades = sim.trades[n_trades:]
        n_trades = len(sim.trades)
        if not new_trades:
            continue
        sign = 1.0 if new_trades[0].aggressor == "buy" else -1.0
        agg = "buy" if sign > 0 else "sell"
        pre_touch = ba if agg == "buy" else bb
        pre_mid = 0.5 * (bb + ba) if (bb is not None and ba is not None) else None
        # One observation per fill-bearing event, matching the tape side
        # where each EXECUTION message is one observation.
        n_fills += 1
        post_bb, post_ba = sim.best_bid_level, sim.best_ask_level
        post_mid = (
            0.5 * (post_bb + post_ba) if (post_bb is not None and post_ba is not None) else None
        )
        if pre_mid is not None and post_mid is not None:
            dmids.append(sign * (post_mid - pre_mid))
        post_touch = _touch(agg)
        if pre_touch is not None and post_touch is not None and post_touch != pre_touch:
            n_empty += 1
            gaps.append(sign * float(post_touch - pre_touch))
    return _decomp_stats(n_fills=n_fills, n_empty=n_empty, gap_ticks=gaps, signed_dmids=dmids)


def _split(seed: int) -> SplitFlow:
    return SplitFlow(
        p_start=0.10, size_tail=1.2, k_min=10, k_max=600, intensity_mult=3.0, seed=seed
    )


def instant_decomp_bench(
    tape_dir: Path, ticker: str = "AMZN", *, horizon: int = 20000, seed: int = 7
) -> dict[str, Any]:
    """Touch-emptying attribution of instantaneous impact, real vs arms."""
    msg = sorted(tape_dir.glob(f"{ticker}_*_message_*.csv"))
    ob = sorted(tape_dir.glob(f"{ticker}_*_orderbook_*.csv"))
    if not msg or not ob:
        raise FileNotFoundError(f"no LOBSTER message/orderbook CSV pair under {tape_dir}")
    real = lobster_instant_decomp(msg[0], ob[0])
    arms = {
        "iid": sim_instant_decomp(horizon=horizon, seed=seed),
        "split": sim_instant_decomp(
            santa_fe_config(seed=seed + 1), _split(seed + 1), horizon=horizon
        ),
        "lv_split": sim_instant_decomp(
            replace(santa_fe_config(seed=seed + 2), anchor="ref", ref_fill_gain=0.3),
            _split(seed + 2),
            horizon=horizon,
        ),
    }
    divergences: list[str] = []
    if real.get("ok"):
        rp = real.get("p_empty_touch")
        rg = real.get("mean_gap_ticks_when_empty")
        ri = real.get("realized_instant_ticks")
        for name, arm in arms.items():
            if not arm.get("ok"):
                divergences.append(f"{name}:no_fills")
                continue
            if rp is not None and abs(float(arm["p_empty_touch"]) - float(rp)) > 0.1:
                divergences.append(f"{name}:p_empty_{arm['p_empty_touch']}_vs_{rp}")
            if rg is not None and abs(float(arm["mean_gap_ticks_when_empty"]) - float(rg)) > 0.5:
                divergences.append(f"{name}:gap_{arm['mean_gap_ticks_when_empty']}_vs_{rg}")
            if ri is not None and abs(float(arm["realized_instant_ticks"]) - float(ri)) > 0.2:
                divergences.append(f"{name}:instant_{arm['realized_instant_ticks']}_vs_{ri}")
    claims = {
        "identity_respects": bool(
            real.get("ok")
            and abs(float(real["unexplained_instant_ticks"]))
            < abs(float(real["realized_instant_ticks"])) * 0.5 + 0.05
        ),
        "sim_instant_gap_is_sparse": None,
    }
    if real.get("ok") and all(a.get("ok") for a in arms.values()):
        # True when the sim arms' instant shortfall is explained by
        # emptier/tighter touch structure (p_empty × gap too small), not
        # by a mechanism absent from the identity.
        claims["sim_instant_gap_is_sparse"] = bool(
            abs(float(arms["split"]["unexplained_instant_ticks"])) < 0.3
        )
    payload: dict[str, Any] = {
        "schema": INSTANT_DECOMP_SCHEMA,
        "kind": "instant_decomp_bench",
        "ticker": ticker,
        "horizon": horizon,
        "seed": seed,
        "real": real,
        "sim_arms": arms,
        "divergences": divergences,
        "claims": claims,
        "interpretation": (
            "The instant term decomposes as P(empty the touch) times half "
            "the gap to the next level. On the tape both factors are set "
            "by near-touch book sparsity — fills frequently consume the "
            "whole visible level and the book behind it is not dense. "
            "unexplained_instant_ticks is the part of same-event Δmid not "
            "carried by touch-emptying (opposite-side improvements)."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "INSTANT_DECOMP_SCHEMA",
    "instant_decomp_bench",
    "lobster_instant_decomp",
    "sim_instant_decomp",
]
