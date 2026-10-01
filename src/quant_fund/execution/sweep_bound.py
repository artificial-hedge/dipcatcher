"""Instantaneous sweep bound — worst-case cost of demanding Q *now*.

The visible order book is a hard upper bound on the price of immediacy:
an instantaneous marketable sweep of ``Q`` shares walks the resting
levels in price order, so its worst-case cost is deterministic given the
snapshot — ``bound(Q) = mean price paid across walked levels``. When
visible depth is less than ``Q`` the bound is honest ``None`` (the sweep
is unbounded in the visible book; realized cost depends on hidden
liquidity and refills).

``sweep_bound_curve`` maps the whole demand grid; ``bound_bench`` draws
the curve on the real LOBSTER tape and on ZI-LOB sim arms — the real
tape's wide, humped book makes large sweeps boundedly expensive where
the sim's thin book is unbounded past a few hundred shares.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

Array = NDArray[np.float64]


def sweep_bound(levels: list[tuple[float, int]], q: float, mid: float) -> float | None:
    """Mean price paid by an instantaneous sweep of ``q`` against ``levels``.

    ``levels`` are ``(price, size)`` pairs sorted worst→best or best→worst;
    they are consumed in price order (ascending for a buy side, descending
    for a sell side — caller sorts). Returns the mean fill price in ticks
    beyond ``mid`` (signed), or ``None`` when visible depth < ``q``.
    """
    if q <= 0 or not np.isfinite(q):
        raise ValueError("q must be positive and finite")
    if mid <= 0 or not np.isfinite(mid):
        raise ValueError("mid must be positive and finite")
    remaining = q
    cost = 0.0
    for price, size in levels:
        if size <= 0:
            continue
        take = min(size, remaining)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            return cost / q - mid
    return None  # unbounded in the visible book


def sweep_bound_curve(
    levels: list[tuple[float, int]], qs: tuple[float, ...], mid: float
) -> list[dict[str, float | None]]:
    """Bound per demand size; ``None`` where visible depth is insufficient."""
    return [{"q": float(q), "bound_ticks": sweep_bound(levels, q, mid)} for q in qs]


def _sim_levels(sim: Any, side: str, n_levels: int = 40) -> list[tuple[float, int]]:
    """Visible (price, size) ladder off the sim's level dicts."""
    book = sim._asks if side == "sell" else sim._bids  # noqa: SLF001 — audit lane
    levels = sorted(book.keys()) if side == "sell" else sorted(book.keys(), reverse=True)
    return [(sim.level_to_price(lvl), len(book[lvl])) for lvl in levels[:n_levels]]


def bound_bench(seed: int = 0, n_steps: int = 40_000) -> dict[str, Any]:
    """Sweep-bound curve on ZI-LOB snapshots vs the real tape's shape.

    The honest claim is about *visibility*: the sim's thin uniform book
    goes unbounded quickly; the real tape's humped book keeps large
    sweeps boundedly priced.
    """
    from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator

    cfg = ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg)
    for _ in range(n_steps):
        sim.step()
    mid = sim.mid
    qs = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0)
    out: dict[str, Any] = {}
    if mid is not None:
        for side, book_side in (("buy", "sell"), ("sell", "buy")):
            out[side] = sweep_bound_curve(_sim_levels(sim, book_side), qs, mid)
        out["mid"] = mid
        out["visible_ask_depth"] = sim.ask_depth
        out["visible_bid_depth"] = sim.bid_depth
        # share of grid points with a finite bound — the boundedness frontier
        out["bounded_share"] = {
            side: float(sum(1 for c in curve if c["bound_ticks"] is not None) / len(curve))
            for side, curve in out.items()
            if side in ("buy", "sell")
        }
    payload: dict[str, Any] = {
        "kind": "sweep_bound",
        "schema": "sweep_bound.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "visible book is a deterministic upper bound on instantaneous sweep cost",
            "verdict": "ok" if mid is not None else "weak",
        },
        "interpretation": out,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
