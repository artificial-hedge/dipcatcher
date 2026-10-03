"""Empirical birth-death level-depth dynamics on the ZI limit order book.

**Labeled SYNTHETIC** research diagnostics (Cont–Kukanov-style order-flow /
depth dynamics): in the ZI book each price level's queue length is a
birth-death process — limit arrivals add a unit order, market orders and
cancellations remove one. This module measures those dynamics *empirically*
inside ``microstructure.zi_lob_simulator.ZILobSimulator``:

- ``collect_depth_panel`` steps the simulator on the **event clock** (one row
  per processed event — not wall-clock sampling) and records the depth at the
  first ``depth`` absolute price levels inside the touch at panel start, plus
  the running touch. Levels are tracked in *absolute* grid space (the anchor
  frame is frozen at collection start), so every depth change at a tracked
  level is a genuine ±1 event effect attributable to the step's event kind —
  the touch-relative re-indexing contamination of a moving frame never enters.
- ``arrival_cancel_rates`` decomposes the observed changes into birth rates
  (limit arrivals), cancellation death rates, and market-order consumption
  rates per (side, level), with exposure-time denominators (calendar time,
  time-alive, time-at-best, time-in-placement-band).
- ``depletion_time_ctmc`` is the analytic mean hitting time of depth 0 for
  the constant-rate birth-death CTMC — the textbook absorption time, pinned
  against Monte Carlo of the embedded chain in tests.
- ``book_depletion_bench`` seals a ``book_depletion.v1`` SYNTHETIC receipt:
  per-level empirical rates, complete level-depletion spells vs the CTMC
  model (Monte Carlo and analytic), and a bid/ask symmetry check.

Honesty: every output is a SYNTHETIC correctness diagnostic on the simulator,
never market evidence. No Sharpe/P&L keys, ``live_pnl_claim`` is always
``False``, and there is no broker connectivity or live-trading claim.

References:
- Cont, Stoikov, Talreja (2010). A stochastic model for order book dynamics.
  *Operations Research* 58(1):191-205 — ZI-LOB queueing model.
- Cont, Kukanov (2017). Optimal order placement in limit order markets.
  *Quantitative Finance* 17(1):21-39 — order-flow imbalance / depth dynamics.
- Karlin & Taylor (1975). *A First Course in Stochastic Processes* —
  birth-death first-passage (gambler's ruin) mean hitting times.
- Moret & Lillo (2026). arXiv:2609.11614 — Santa Fe calibration the panel
  collector runs on by default.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes
from quant_fund.utils.receipt import seal_receipt

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

BOOK_DEPLETION_KIND = "book_depletion.v1"
INIT_KIND = "init"
EVENT_KINDS = frozenset({"limit", "market", "cancel"})

# Event budget for the initial two-sided-book wait: far beyond anything the
# seeded simulator needs (a handful of limit arrivals always re-seed a side).
_AWAIT_BUDGET = 5_000

# Forbidden headline-metric tokens (mirrors research.catalog / fx1.honesty).
_FORBIDDEN_TOKENS = ("sharpe", "sortino", "calmar", "nav")


# ---------------------------------------------------------------------------
# Fail-closed validation helpers
# ---------------------------------------------------------------------------


def _pos_int(x: int, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise ValueError(f"{name} must be an int >= 1, got {x!r}")
    return x


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


def _await_two_sided(sim: ZILobSimulator) -> None:
    """Step until both book sides are populated (fail-closed on budget)."""
    n_start = sim.n_events
    while sim.best_bid_level is None or sim.best_ask_level is None:
        if sim.n_events - n_start >= _AWAIT_BUDGET:
            raise RuntimeError(
                f"book did not regain two sides within {_AWAIT_BUDGET} events of panel start"
            )
        sim.step()


# ---------------------------------------------------------------------------
# Panel collection (event-clock sampling)
# ---------------------------------------------------------------------------


def collect_depth_panel(sim: ZILobSimulator, *, sample_interval: int, depth: int) -> dict[str, Any]:
    """Step ``sim`` for ``sample_interval`` events, sampling depth each event.

    One panel row is recorded for the initial state (``kind == "init"``) and
    one after every ``sim.step()`` — event-clock sampling, not wall-clock —
    giving ``sample_interval + 1`` rows. The ``depth`` tracked levels per side
    are the absolute grid levels inside the touch **at panel start**
    (``bid_levels[i] = best_bid_level - i``, ``ask_levels[i] =
    best_ask_level + i``); the frame never re-indexes when the touch moves,
    so tracked deltas are genuine event effects.

    Returns a dict of arrays: ``t``, ``n_events``, ``kind``, ``mid``,
    ``best_bid`` / ``best_ask`` / ``best_bid_level`` / ``best_ask_level``
    (NaN while that side is empty), ``bid_depth`` / ``ask_depth``
    (``[n_rows, depth]`` unit-order counts), plus the ``*_levels`` anchors and
    the sim config echoes needed downstream (``band``, ``tick``, ``s0``).
    Fail-closed: wrong sim type, ``sample_interval < 1``, ``depth < 1``.
    """
    if not isinstance(sim, ZILobSimulator):
        raise TypeError(f"sim must be a ZILobSimulator, got {type(sim).__name__}")
    n_steps = _pos_int(sample_interval, "sample_interval")
    d = _pos_int(depth, "depth")
    _await_two_sided(sim)
    bb0 = sim.best_bid_level
    ba0 = sim.best_ask_level
    if bb0 is None or ba0 is None:  # pragma: no cover - _await_two_sided raised
        raise RuntimeError("book lacks two sides after await")
    bid_levels = np.arange(bb0, bb0 - d, -1, dtype=np.int64)
    ask_levels = np.arange(ba0, ba0 + d, 1, dtype=np.int64)

    n_rows = n_steps + 1
    t_arr = np.empty(n_rows, dtype=np.float64)
    ev_arr = np.empty(n_rows, dtype=np.int64)
    kind_arr = np.empty(n_rows, dtype="<U6")
    mid_arr = np.empty(n_rows, dtype=np.float64)
    bb_price = np.empty(n_rows, dtype=np.float64)
    ba_price = np.empty(n_rows, dtype=np.float64)
    bb_level = np.empty(n_rows, dtype=np.float64)
    ba_level = np.empty(n_rows, dtype=np.float64)
    bid_depth = np.empty((n_rows, d), dtype=np.int64)
    ask_depth = np.empty((n_rows, d), dtype=np.int64)

    def _record(r: int, kind: str) -> None:
        t_arr[r] = sim.t
        ev_arr[r] = sim.n_events
        kind_arr[r] = kind
        mid = sim.mid
        mid_arr[r] = mid if mid is not None else math.nan
        bb, ba = sim.best_bid, sim.best_ask
        bb_price[r] = bb if bb is not None else math.nan
        ba_price[r] = ba if ba is not None else math.nan
        bl, al = sim.best_bid_level, sim.best_ask_level
        bb_level[r] = float(bl) if bl is not None else math.nan
        ba_level[r] = float(al) if al is not None else math.nan
        bid_depth[r] = [sim.depth_at("buy", int(lvl)) for lvl in bid_levels]
        ask_depth[r] = [sim.depth_at("sell", int(lvl)) for lvl in ask_levels]

    _record(0, INIT_KIND)
    for r in range(1, n_rows):
        _record(r, sim.step())
    return {
        "t": t_arr,
        "n_events": ev_arr,
        "kind": kind_arr,
        "mid": mid_arr,
        "best_bid": bb_price,
        "best_ask": ba_price,
        "best_bid_level": bb_level,
        "best_ask_level": ba_level,
        "bid_levels": bid_levels,
        "ask_levels": ask_levels,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "n_rows": n_rows,
        "depth": d,
        "band": int(sim.cfg.band),
        "tick": float(sim.cfg.tick),
        "s0": float(sim.cfg.s0),
        "seed": int(sim.cfg.seed),
    }


def _panel_field(panel: Mapping[str, Any], name: str) -> Any:
    if not isinstance(panel, Mapping):
        raise TypeError(f"panel must be a mapping, got {type(panel).__name__}")
    if name not in panel:
        raise ValueError(f"panel is missing required field {name!r}")
    return panel[name]


def _side_rates(
    side: Literal["bid", "ask"],
    depth_path: IntArray,
    levels: IntArray,
    own_best: Array,
    opp_best: Array,
    kinds: NDArray[np.str_],
    dt: Array,
    band: int,
) -> dict[str, Any]:
    """Per-level birth/cancel/market-order rates for one book side.

    ``depth_path[:, i]`` is the depth path of absolute level ``levels[i]``;
    ``kinds[r]`` is the event kind of the step from row ``r`` to ``r+1`` and
    ``dt[r]`` its holding time. A ``+1`` change at a tracked level is a limit
    arrival (birth), a ``-1`` under a ``cancel`` step is a cancellation and
    under a ``market`` step a consumption at the touch — any other change
    pattern violates the simulator's unit-event invariant and raises.

    Rate denominators are exposure times: births per calendar second,
    ``birth_rate_in_band`` per second the level was inside the limit-order
    placement band of the opposite touch, deaths per second the level was
    nonempty, ``market_rate_at_best`` per second the level sat at the touch
    with depth > 0, and ``death_rate_per_unit`` per unit-depth second
    (the per-order hazard → the cancel intensity under per-order clocks).
    """
    n_rows, d = depth_path.shape
    n_intervals = n_rows - 1
    prev = depth_path[:-1]
    delta = np.diff(depth_path, axis=0)
    if np.abs(delta).max() > 1:
        raise ValueError("panel violates the unit-event invariant (|depth delta| > 1)")
    born = delta == 1
    died = delta == -1
    step_kind_grid = np.broadcast_to(kinds[:, None], delta.shape)
    if bool((step_kind_grid[born] != "limit").any()):
        raise ValueError("panel shows a birth under a non-limit event kind")
    if bool((~np.isin(step_kind_grid[died], ["market", "cancel"])).any()):
        raise ValueError("panel shows a death under a non-removal event kind")

    n_birth = born.sum(axis=0).astype(np.float64)
    death_mo = died & (kinds == "market")[:, None]
    death_cxl = died & (kinds == "cancel")[:, None]
    n_death_mo = death_mo.sum(axis=0).astype(np.float64)
    n_death_cxl = death_cxl.sum(axis=0).astype(np.float64)
    n_death = n_death_mo + n_death_cxl

    window_s = float(dt.sum())
    alive_prev = prev > 0
    alive_s = (dt[:, None] * alive_prev).sum(axis=0)
    unit_s = (dt[:, None] * prev).sum(axis=0)
    # A level receives limit flow only while it lies inside the placement
    # band of the opposite touch: bids at opp_best - [1, band], asks at
    # opp_best + [1, band]. NaN opposite bests mark the exposure unmeasured.
    opp_prev = opp_best[:-1, None]
    lvl = levels[None, :].astype(np.float64)
    if side == "bid":
        in_band_prev = (lvl >= opp_prev - float(band)) & (lvl <= opp_prev - 1.0)
    else:
        in_band_prev = (lvl >= opp_prev + 1.0) & (lvl <= opp_prev + float(band))
    in_band_s = (dt[:, None] * in_band_prev).sum(axis=0)
    at_best_prev = alive_prev & np.isfinite(own_best[:-1])[:, None] & (lvl == own_best[:-1, None])
    at_best_s = (dt[:, None] * at_best_prev).sum(axis=0)

    def _rate(counts: NDArray[np.float64], exposure: Array) -> Array:
        out = np.full(d, np.nan, dtype=np.float64)
        ok = exposure > 0.0
        out[ok] = counts[ok] / exposure[ok]
        return out

    return {
        "levels": levels.astype(np.int64),
        "n_birth": n_birth,
        "n_death_cancel": n_death_cxl,
        "n_death_market": n_death_mo,
        "n_death": n_death,
        "window_s": window_s,
        "alive_s": alive_s,
        "in_band_s": in_band_s,
        "at_best_s": at_best_s,
        "unit_depth_s": unit_s,
        "mean_depth": unit_s / window_s,
        "birth_rate": n_birth / window_s,
        "birth_rate_in_band": _rate(n_birth, in_band_s),
        "death_rate": _rate(n_death, alive_s),
        "cancel_rate": _rate(n_death_cxl, alive_s),
        "market_rate": _rate(n_death_mo, alive_s),
        "market_rate_at_best": _rate(n_death_mo, at_best_s),
        "death_rate_per_unit": _rate(n_death, unit_s),
        "n_intervals": n_intervals,
    }


def arrival_cancel_rates(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Empirical per-(side, level) birth/death rates from a depth panel.

    Fail-closed: not a panel mapping, missing fields, fewer than two rows, a
    non-positive window, or a depth path that violates the simulator's
    unit-event invariant (``|Δdepth| > 1`` or a sign/kind mismatch) raise.
    Levels with zero relevant exposure report ``NaN`` rates — an honest
    unmeasured value, never a fabricated zero.
    """
    t = np.asarray(_panel_field(panel, "t"), dtype=np.float64)
    kinds = np.asarray(_panel_field(panel, "kind"), dtype=np.str_)
    bid_depth = np.asarray(_panel_field(panel, "bid_depth"), dtype=np.int64)
    ask_depth = np.asarray(_panel_field(panel, "ask_depth"), dtype=np.int64)
    bid_levels = np.asarray(_panel_field(panel, "bid_levels"), dtype=np.int64)
    ask_levels = np.asarray(_panel_field(panel, "ask_levels"), dtype=np.int64)
    own_bb = np.asarray(_panel_field(panel, "best_bid_level"), dtype=np.float64)
    own_ba = np.asarray(_panel_field(panel, "best_ask_level"), dtype=np.float64)
    band = int(_panel_field(panel, "band"))
    if t.size < 2:
        raise ValueError(f"panel needs >= 2 rows, got {t.size}")
    dt = np.diff(t)
    if not np.all(dt >= 0.0):
        raise ValueError("panel timestamps are not non-decreasing")
    window_s = float(dt.sum())
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError(f"panel window must be positive, got {window_s!r}")
    if not np.all(np.isin(kinds[1:], list(EVENT_KINDS))):
        raise ValueError("panel carries an unknown event kind")
    n_rows = int(t.size)
    for name, arr in (("bid_depth", bid_depth), ("ask_depth", ask_depth)):
        if arr.shape[0] != n_rows:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n_rows}")
    step_kinds = np.asarray(kinds[1:], dtype=np.str_)
    return {
        "kind": "arrival_cancel_rates.v1",
        "label": "SYNTHETIC",
        "window_s": window_s,
        "n_rows": n_rows,
        "n_events": n_rows - 1,
        "depth": int(_panel_field(panel, "depth")),
        "band": band,
        "bid": _side_rates("bid", bid_depth, bid_levels, own_bb, own_ba, step_kinds, dt, band),
        "ask": _side_rates("ask", ask_depth, ask_levels, own_ba, own_bb, step_kinds, dt, band),
    }


# ---------------------------------------------------------------------------
# Level-depletion spells (observed hitting events)
# ---------------------------------------------------------------------------


def level_depletion_spells(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Complete nonempty spells per tracked level — the observed depletions.

    A spell starts at the row where a level's depth rises 0 -> 1 (a birth) and
    ends at the row where it falls to 0; ``q_start`` is the depth right after
    the birth (always 1 under unit events) and ``duration_s`` the sim-time
    span. Spells open at the first row or still open at the last are
    left/right-censored — counted and excluded, never silently truncated.
    """
    t = np.asarray(_panel_field(panel, "t"), dtype=np.float64)
    if t.size < 2:
        raise ValueError(f"panel needs >= 2 rows, got {t.size}")
    out_sides: dict[str, Any] = {}
    n_left = n_right = 0
    for side in ("bid", "ask"):
        path = np.asarray(_panel_field(panel, f"{side}_depth"), dtype=np.int64)
        spells_per_level: list[list[dict[str, Any]]] = []
        for i in range(path.shape[1]):
            q = path[:, i]
            spells: list[dict[str, Any]] = []
            r = 0
            n = q.size
            while r < n:
                if q[r] <= 0:
                    r += 1
                    continue
                start = r
                if start == 0:
                    n_left += 1
                while r < n and q[r] > 0:
                    r += 1
                if r < n:
                    if start > 0:
                        spells.append(
                            {
                                "q_start": int(q[start]),
                                "duration_s": float(t[r] - t[start]),
                            }
                        )
                else:
                    n_right += 1 if start > 0 else 0
            spells_per_level.append(spells)
        out_sides[side] = spells_per_level
    n_complete = sum(len(sp) for spells in out_sides.values() for sp in spells)
    return {
        "bid": out_sides["bid"],
        "ask": out_sides["ask"],
        "n_complete": n_complete,
        "n_left_censored": n_left,
        "n_right_censored": n_right,
    }


# ---------------------------------------------------------------------------
# Analytic CTMC hitting time
# ---------------------------------------------------------------------------


def depletion_time_ctmc(q0: int, lam_birth: float, mu_death: float) -> float:
    """Exact mean time to hit depth 0 for the constant-rate birth-death CTMC.

    The chain moves ``q -> q + 1`` at rate ``lam_birth`` and ``q -> q - 1`` at
    rate ``mu_death`` from every state ``q >= 1``; 0 is absorbing. The
    first-step equations on the embedded chain are

        E_0 = 0,
        E_i = 1/(lam+mu) + lam/(lam+mu) * E_{i+1} + mu/(lam+mu) * E_{i-1},

    whose textbook recurrence-sum solution (Karlin & Taylor),

        E_i = sum_{j=1..i} (1/mu) * sum_{k>=j} prod_{l=j+1..k} (lam/mu),

    telescopes for constant rates into the geometric-series closed form

        E_i = i / (mu - lam)     when lam < mu,

        = +inf                   when lam >= mu (positive/zero drift — the
                                 walk is not almost-surely absorbed, and at
                                 lam == mu absorption is certain but the
                                 mean diverges).
    ``lam == 0`` is the pure-death chain: ``q0`` sequential ``Exp(mu)``
    holding times, mean ``q0 / mu`` — recovered by the same formula.

    Fail-closed: ``q0 < 1``, ``lam_birth < 0``, or ``mu_death <= 0`` (and any
    non-finite rate) raise ``ValueError``.
    """
    q = _pos_int(q0, "q0")
    lam = _nonneg_finite(lam_birth, "lam_birth")
    mu = _pos_finite(mu_death, "mu_death")
    if lam >= mu:
        return math.inf
    return float(q) / (mu - lam)


def _mc_depletion_mean(
    q0: int, lam: float, mu: float, *, n_paths: int, rng: np.random.Generator, step_cap: int
) -> tuple[float, int]:
    """Vectorized Monte Carlo mean hitting time of the constant-rate CTMC.

    Gillespie-exact embedded chain: each step draws an ``Exp(lam+mu)``
    holding time and a ±1 move with ``P(up) = lam/(lam+mu)``. Paths still
    alive after ``step_cap`` jumps are right-censored and excluded from the
    mean; the returned count lets the receipt report the censoring. With
    ``lam == 0`` this is a pure-death chain: exactly ``q0`` steps, all
    completed.
    """
    rate = lam + mu
    p_up = lam / rate
    q = np.full(n_paths, q0, dtype=np.int64)
    t = np.zeros(n_paths, dtype=np.float64)
    alive = np.ones(n_paths, dtype=np.bool_)
    steps = 0
    while bool(alive.any()) and steps < step_cap:
        t += np.where(alive, rng.exponential(1.0 / rate, n_paths), 0.0)
        move = np.where(rng.random(n_paths) < p_up, 1, -1)
        q = q + np.where(alive, move, 0)
        alive = q > 0
        steps += 1
    done = t[~alive]
    mean = float(done.mean()) if done.size else math.nan
    return mean, int(alive.sum())


# ---------------------------------------------------------------------------
# Sealed bench receipt
# ---------------------------------------------------------------------------


def _rel_err(model: float, observed: float) -> float:
    if not (math.isfinite(model) and math.isfinite(observed)) or observed <= 0.0:
        return math.nan
    return abs(model - observed) / observed


def _rel_diff(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    denom = 0.5 * (a + b)
    if denom <= 0.0:
        return math.nan
    return abs(a - b) / denom


def _nanmean_list(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.mean(finite)) if finite else math.nan


def book_depletion_bench(
    *,
    config: ZILobConfig | None = None,
    n_events: int = 4_000,
    depth: int = 3,
    mc_paths: int = 64,
    mc_step_cap: int = 20_000,
    sym_tol: float = 0.35,
    seed: int = 0,
) -> dict[str, Any]:
    """Sealed ``book_depletion.v1`` receipt (SYNTHETIC): level dynamics vs CTMC.

    Runs the ZI simulator (``config`` or the Santa Fe calibration seeded by
    ``seed``), collects the event-clock depth panel, estimates per-level
    birth/death rates, then scores the constant-rate birth-death model against
    the observed level-depletion spells: for every (side, level) with at
    least one complete spell, a seeded Monte Carlo of the fitted CTMC predicts
    the mean depletion time and is compared against the observed mean;
    ``mean_rel_err`` aggregates the per-level relative errors. The analytic
    closed form is reported alongside, and a bid/ask symmetry check compares
    pooled side rates at tolerance ``sym_tol``.

    Fail-closed on bad parameters and on degenerate panels (``n_events < 1``,
    ``depth < 1``, ``mc_paths < 1``, ``mc_step_cap < 1``, ``sym_tol <= 0``).
    Deterministic: identical ``(seed, params)`` give identical receipts,
    including ``receipt_sha256``.
    """
    _pos_int(n_events, "n_events")
    d = _pos_int(depth, "depth")
    _pos_int(mc_paths, "mc_paths")
    _pos_int(mc_step_cap, "mc_step_cap")
    tol = _pos_finite(sym_tol, "sym_tol")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    cfg = config if config is not None else santa_fe_config(seed=seed)
    if not isinstance(cfg, ZILobConfig):
        raise TypeError(f"config must be a ZILobConfig, got {type(cfg).__name__}")

    sim = ZILobSimulator(cfg)
    panel = collect_depth_panel(sim, sample_interval=n_events, depth=d)
    rates = arrival_cancel_rates(panel)
    spells = level_depletion_spells(panel)

    per_level: dict[str, Any] = {}
    level_rel_mc: list[float] = []
    level_rel_analytic: list[float] = []
    n_scored = 0
    n_censored_paths = 0
    for side_idx, side in enumerate(("bid", "ask")):
        side_rates = rates[side]
        rows: list[dict[str, Any]] = []
        for i in range(d):
            lam_hat = float(side_rates["birth_rate"][i])
            mu_hat = float(side_rates["death_rate"][i])
            level_spells = spells[side][i]
            obs_mean = (
                float(np.mean([s["duration_s"] for s in level_spells]))
                if level_spells
                else math.nan
            )
            mc_mean = math.nan
            analytic = math.nan
            rel_mc = math.nan
            rel_analytic = math.nan
            if level_spells and math.isfinite(mu_hat) and mu_hat > 0.0:
                # All complete spells here start at q_start == 1 under the
                # unit-event invariant; group the unique entry depths so one
                # seeded MC batch prices them all.
                for group, q_start in enumerate(sorted({s["q_start"] for s in level_spells})):
                    rng = np.random.default_rng(seed * 1_000_003 + side_idx * 977 + i * 131 + group)
                    mc_g, n_cens = _mc_depletion_mean(
                        q_start,
                        lam_hat,
                        mu_hat,
                        n_paths=mc_paths,
                        rng=rng,
                        step_cap=mc_step_cap,
                    )
                    n_censored_paths += n_cens
                    if group == 0:
                        mc_mean = mc_g
                analytic = depletion_time_ctmc(1, lam_hat, mu_hat)
                rel_mc = _rel_err(mc_mean, obs_mean)
                rel_analytic = _rel_err(analytic, obs_mean)
                n_scored += 1
                level_rel_mc.append(rel_mc)
                level_rel_analytic.append(rel_analytic)
            rows.append(
                {
                    "level_index": i,
                    "level": int(side_rates["levels"][i]),
                    "birth_rate": lam_hat,
                    "death_rate": mu_hat,
                    "cancel_rate": float(side_rates["cancel_rate"][i]),
                    "market_rate_at_best": float(side_rates["market_rate_at_best"][i]),
                    "death_rate_per_unit": float(side_rates["death_rate_per_unit"][i]),
                    "birth_rate_in_band": float(side_rates["birth_rate_in_band"][i]),
                    "mean_depth": float(side_rates["mean_depth"][i]),
                    "n_birth": int(side_rates["n_birth"][i]),
                    "n_death": int(side_rates["n_death"][i]),
                    "n_spells": len(level_spells),
                    "obs_mean_depletion_s": obs_mean,
                    "mc_mean_depletion_s": mc_mean,
                    "analytic_depletion_s": analytic,
                    "rel_err_mc": rel_mc,
                    "rel_err_analytic": rel_analytic,
                }
            )
        per_level[side] = rows

    bid_birth = _nanmean_list([float(v) for v in rates["bid"]["birth_rate"]])
    ask_birth = _nanmean_list([float(v) for v in rates["ask"]["birth_rate"]])
    bid_death = _nanmean_list([float(v) for v in rates["bid"]["death_rate"]])
    ask_death = _nanmean_list([float(v) for v in rates["ask"]["death_rate"]])
    sym_birth = _rel_diff(bid_birth, ask_birth)
    sym_death = _rel_diff(bid_death, ask_death)
    symmetric = bool(
        math.isfinite(sym_birth)
        and math.isfinite(sym_death)
        and sym_birth <= tol
        and sym_death <= tol
    )

    receipt: dict[str, Any] = {
        "schema": BOOK_DEPLETION_KIND,
        "kind": BOOK_DEPLETION_KIND,
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "model": {
            "type": "constant_rate_birth_death_ctmc",
            "hitting_time_formula": "E[T_q] = q/(mu-lam) for lam<mu else +inf",
            "rate_estimator": (
                "births per window second; deaths per nonempty second; "
                "market deaths additionally per at-touch second"
            ),
        },
        "params": {
            "n_events": int(n_events),
            "depth": d,
            "mc_paths": int(mc_paths),
            "mc_step_cap": int(mc_step_cap),
            "sym_tol": tol,
            "seed": int(seed),
            "config": {
                "s0": float(cfg.s0),
                "tick": float(cfg.tick),
                "lam": float(cfg.lam),
                "mu": float(cfg.mu),
                "theta_cxl": float(cfg.theta_cxl),
                "p_buy": float(cfg.p_buy),
                "band": int(cfg.band),
                "density_exponent": float(cfg.density_exponent),
                "anchor": str(cfg.anchor),
                "seed": int(cfg.seed),
            },
        },
        "window_s": float(rates["window_s"]),
        "n_rows": int(rates["n_rows"]),
        "n_events_processed": int(rates["n_events"]),
        "per_level": per_level,
        "spells": {
            "n_complete": int(spells["n_complete"]),
            "n_left_censored": int(spells["n_left_censored"]),
            "n_right_censored": int(spells["n_right_censored"]),
        },
        "agreement": {
            "n_levels_scored": n_scored,
            "n_levels": 2 * d,
            "mc_rel_err_mean": _nanmean_list(level_rel_mc),
            "analytic_rel_err_mean": _nanmean_list(level_rel_analytic),
            "n_mc_paths_per_level": int(mc_paths),
            "n_mc_paths_censored": n_censored_paths,
        },
        "symmetry": {
            "bid_birth_rate_mean": bid_birth,
            "ask_birth_rate_mean": ask_birth,
            "birth_rel_diff": sym_birth,
            "bid_death_rate_mean": bid_death,
            "ask_death_rate_mean": ask_death,
            "death_rel_diff": sym_death,
            "tol": tol,
            "sides_symmetric": symmetric,
        },
        "event_counts": sim.event_counts(),
    }
    leaked = [k for k in receipt if any(tok in k.lower() for tok in _FORBIDDEN_TOKENS)]
    if leaked:
        raise AssertionError(f"book_depletion bench leaked forbidden research keys: {leaked}")
    # Canonicalize to JSON-native values before sealing: non-finite floats
    # become null (honest unmeasured values) and the sealed receipt compares
    # equal across identical seeded runs.
    return seal_receipt(json.loads(canonical_json_bytes(receipt)))
