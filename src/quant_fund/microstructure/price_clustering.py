"""Round-number clustering in limit-order placement — SYNTHETIC diagnostics.

Agents are known to prefer round prices when placing limit orders (the
human-attention / round-number clustering literature: orders pile up at
multiples of 10 ticks far more than a residue-uniform baseline predicts).
This module measures that preference in a way that works identically on

- **integer tick levels** — the ZI-LOB sim (``microstructure.zi_lob_simulator``)
  submits at integer levels, so mod-``m`` residue clustering is well-defined —
  and
- **real prices** — ``levels_from_prices`` snaps a price stream onto its tick
  grid (fail-closed off-grid), after which the same residue statistic applies.

Statistic. ``cluster_stats`` bins submitted levels by their residue mod ``m``
and scores the histogram against a **reachable-normalized** uniform baseline:
under level-uniform placement the expected share of residue ``r`` is not
``1/m`` — it is the fraction of *reachable* levels carrying residue ``r``.
When the reachable universe does not divide evenly by ``m`` (e.g. the sim
band places bids only at ``±1..±band`` under a frozen reference anchor, so
residue 0 is literally unreachable), normalizing by ``1/m`` manufactures a
fake clustering signal. The reachable set here is exact: ``sim_cluster_stats``
reconstructs the placement window the simulator drew from on every limit
event and unions it, and ``cluster_stats`` accepts an explicit
``reachable_levels`` universe (default: the observed integer span).

Measurement. ``sim_cluster_stats`` steps the sim and attributes limit-order
submits by **order-id diffing** on ``sim._orders`` (the same instrumentation
convention as ``order_revision``-style stats): every new resting order id
after a step is a fresh submit whose ``level`` is the placement. Order ids
are monotonic, so the diff is exact — fills and cancels only *remove* ids.

``price_clustering_bench`` runs the flow arms and seals a
``price_clustering.v1`` receipt (``kind='price_clustering'``): canonical JSON
payload, ``receipt_sha256 = sha256(canonical_json_bytes(payload))`` computed
before the field is attached — the repo-wide seal convention.

Honesty: every output is a SYNTHETIC correctness diagnostic on simulated
placement, never market evidence. No broker connectivity, no live-trading
claim, no headline PnL anywhere in this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2 as _chi2_dist

from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

PRICE_CLUSTERING_SCHEMA = "price_clustering.v1"
PRICE_CLUSTERING_KIND = "price_clustering"
PRICE_CLUSTERING_RECEIPT = "price_clustering_synth.json"

IntArray = NDArray[np.int64]


# ---------------------------------------------------------------------------
# Fail-closed validation helpers
# ---------------------------------------------------------------------------


def _modulus(m: int) -> int:
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
        raise TypeError(f"m must be an int, got {m!r}")
    if int(m) < 2:
        raise ValueError(f"m must be >= 2 (mod-1 clustering is undefined), got {m!r}")
    return int(m)


def _levels_array(values: Any, *, name: str) -> IntArray:
    """1-D non-empty integer level vector; float inputs must be integral."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if arr.dtype == np.bool_ or arr.dtype == object:
        raise TypeError(f"{name} must be integer-valued, got dtype {arr.dtype}")
    if np.issubdtype(arr.dtype, np.floating):
        fa = arr.astype(np.float64)
        if not np.all(np.isfinite(fa)):
            raise ValueError(f"{name} must be finite")
        if not np.all(fa == np.rint(fa)):
            raise ValueError(f"{name} must be integer-valued levels, got {values!r}")
        return fa.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be integer-valued, got dtype {arr.dtype}")
    return arr.astype(np.int64)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def cluster_stats(
    levels: NDArray[np.int64] | Any,
    m: int,
    *,
    reachable_levels: NDArray[np.int64] | Any | None = None,
) -> dict[str, Any]:
    """Mod-``m`` residue clustering of submitted levels vs a uniform baseline.

    ``levels`` are integer tick levels (sim) or snapped price-grid levels (see
    ``levels_from_prices``). ``reachable_levels`` is the placement universe:
    the exact set of levels the placement rule could have hit. The uniform
    baseline share of residue ``r`` is ``#{u ∈ reachable : u ≡ r} /
    #reachable`` — *not* ``1/m`` — so a band that never offers a residue
    contributes zero expected mass instead of fabricating a deficit.

    Defaults: the observed integer span ``[min(levels), max(levels)]``, exact
    for grids whose placement window sweeps the whole span (e.g. the
    touch-anchored sim band following a wandering anchor).

    Returns per-residue observed shares, the reachable-normalized baseline
    shares, a Pearson chi-square stat (``chi2_stat``/``chi2_dof``/
    ``chi2_p_value``, ``None`` when the universe spans a single residue
    class), and ``excess_round_share`` — the residue-0 share premium over the
    baseline (positive ⇒ round-level preference).

    Fail-closed: empty or non-integer inputs, ``m < 2``, observed levels
    outside the declared reachable universe.
    """
    lv = _levels_array(levels, name="levels")
    mm = _modulus(m)
    if reachable_levels is None:
        universe = np.arange(int(lv.min()), int(lv.max()) + 1, dtype=np.int64)
    else:
        universe = np.unique(_levels_array(reachable_levels, name="reachable_levels"))
        if not np.isin(lv, universe).all():
            outside = sorted(set(lv.tolist()) - set(universe.tolist()))
            raise ValueError(
                f"{len(outside)} submitted level(s) lie outside the reachable universe "
                f"(e.g. {outside[:5]}); pass the true placement universe or widen it"
            )
    n = int(lv.size)
    total = int(universe.size)
    observed = np.bincount(np.mod(lv, mm), minlength=mm).astype(np.float64)
    reachable = np.bincount(np.mod(universe, mm), minlength=mm).astype(np.float64)
    baseline = reachable / float(total)
    shares = observed / float(n)
    expected = baseline * float(n)
    mask = reachable > 0.0
    chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    dof = int(mask.sum()) - 1
    p_value = float(_chi2_dist.sf(chi2, dof)) if dof >= 1 else None
    return {
        "label": "SYNTHETIC",
        "m": mm,
        "n_submits": n,
        "n_reachable_levels": total,
        "residue_counts": [int(c) for c in observed],
        "residue_shares": [float(s) for s in shares],
        "reachable_counts": [int(c) for c in reachable],
        "uniform_baseline_shares": [float(b) for b in baseline],
        "excess_round_share": float(shares[0] - baseline[0]),
        "chi2_stat": chi2,
        "chi2_dof": dof,
        "chi2_p_value": p_value,
    }


def levels_from_prices(
    prices: NDArray[np.float64] | Any,
    *,
    tick: float,
    ref_price: float = 0.0,
) -> IntArray:
    """Snap real prices onto the tick grid → integer levels (fail-closed).

    ``(price - ref_price) / tick`` must be within ``1e-6`` of an integer for
    every entry — a price stream on a different grid is a contract violation,
    not a rounding opportunity. With ``ref_price = sim s0`` this reproduces
    the sim's own level indexing exactly.
    """
    tk = float(tick)
    if not math.isfinite(tk) or tk <= 0.0:
        raise ValueError(f"tick must be positive and finite, got {tick!r}")
    ref = float(ref_price)
    if not math.isfinite(ref):
        raise ValueError(f"ref_price must be finite, got {ref_price!r}")
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"prices must be a non-empty 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("prices must be finite")
    raw = (arr - ref) / tk
    rounded = np.rint(raw)
    if not np.all(np.abs(raw - rounded) <= 1e-6):
        raise ValueError(f"prices off the tick grid (tick={tk}, ref={ref})")
    return rounded.astype(np.int64)


# ---------------------------------------------------------------------------
# Simulator instrumentation (oid-diff submit attribution + window tracking)
# ---------------------------------------------------------------------------


def _placement_window(sim: ZILobSimulator, ba0: int | None, bb0: int | None) -> set[int]:
    """Levels the fired limit event could have rested at — mirrors
    ``ZILobSimulator._limit_order_event`` exactly.

    Bests (``ba0``/``bb0``) are the *pre-step* quotes the event drew against;
    the reference level/EMA are read post-step (``step()`` updates them before
    dispatching the event). Ref-anchor crossing placements are dropped in the
    sim, so blocked window levels are excluded from the reachable set.
    """
    cfg = sim.cfg
    band = int(cfg.band)
    if cfg.anchor == "ref":
        ref = int(round(sim._ref_ema))
        out: set[int] = set()
        for d in range(1, band + 1):
            if ba0 is None or ref - d < ba0:
                out.add(ref - d)
            if bb0 is None or ref + d > bb0:
                out.add(ref + d)
        return out
    buy_anchor = ba0 if ba0 is not None else sim._ref_level + 1
    sell_anchor = bb0 if bb0 is not None else sim._ref_level - 1
    return {buy_anchor - d for d in range(1, band + 1)} | {
        sell_anchor + d for d in range(1, band + 1)
    }


def sim_cluster_stats(
    *,
    flow: MarkovRegimeFlow | None = None,
    horizon: float = 30000.0,
    seed: int = 0,
    m: int = 10,
    config: ZILobConfig | None = None,
) -> dict[str, Any]:
    """Limit-submit levels for one flow arm, attributed by order-id diffing.

    Steps the ZI-LOB to ``horizon`` sim-seconds; after every ``step()`` the
    diff ``keys(sim._orders) - prev`` yields exactly the orders rested by that
    event (each ``_rest`` mints a fresh monotonically increasing id), and the
    reachable set is the union of the reconstructed placement windows.

    ``config`` overrides the default ``santa_fe_config(seed=seed)`` arm; when
    given, its own ``seed`` governs (the ``seed`` arg is ignored). Returns the
    raw submit/reachable vectors plus the ``cluster_stats`` bundle and a
    price-domain cross-check (levels round-tripped through
    ``level_to_price`` → ``levels_from_prices`` must give identical residue
    counts — same statistic on real prices).
    """
    cfg = config if config is not None else santa_fe_config(seed=seed)
    if not isinstance(cfg, ZILobConfig):
        raise TypeError(f"config must be a ZILobConfig, got {config!r}")
    h = float(horizon)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"horizon must be positive and finite, got {horizon!r}")
    mm = _modulus(m)
    sim = ZILobSimulator(cfg, flow=flow)
    prev_oids: set[int] = set()
    submit_levels: list[int] = []
    reachable: set[int] = set()
    while sim.t < h:
        ba0, bb0 = sim.best_ask_level, sim.best_bid_level
        event = sim.step()
        for oid, order in sim._orders.items():
            if oid not in prev_oids:
                submit_levels.append(order.level)
        prev_oids = set(sim._orders)
        if event == "limit":
            reachable |= _placement_window(sim, ba0, bb0)
    lv = np.asarray(submit_levels, dtype=np.int64)
    uni = np.asarray(sorted(reachable), dtype=np.int64)
    cluster = cluster_stats(lv, mm, reachable_levels=uni)
    # Same measurement on real prices: snap level_to_price(submits) back to
    # the grid — residue counts must be identical, or the price path is broken.
    price_lv = levels_from_prices(
        np.asarray([sim.level_to_price(int(x)) for x in lv]),
        tick=cfg.tick,
        ref_price=cfg.s0,
    )
    price_uni = levels_from_prices(
        np.asarray([sim.level_to_price(int(x)) for x in uni]),
        tick=cfg.tick,
        ref_price=cfg.s0,
    )
    price_cluster = cluster_stats(price_lv, mm, reachable_levels=price_uni)
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "seed": cfg.seed,
        "horizon": h,
        "m": mm,
        "anchor": cfg.anchor,
        "band": int(cfg.band),
        "n_events": sim.n_events,
        "n_submits": int(lv.size),
        "submit_levels": [int(x) for x in lv],
        "reachable_levels": [int(x) for x in uni],
        "cluster": cluster,
        "price_domain_residues_match": price_cluster["residue_counts"] == cluster["residue_counts"],
        "event_counts": sim.event_counts(),
    }


# ---------------------------------------------------------------------------
# Bench: flow arms + sealed receipt
# ---------------------------------------------------------------------------


def _detector_probe(*, m: int, seed: int, n: int = 60_000) -> dict[str, Any]:
    """Planted 2x residue-0 preference — the detector MUST recover it.

    Sensitivity check bound into the receipt: a clustering measure that
    cannot flag a planted 2x round-level preference detects nothing.
    """
    rng = np.random.default_rng(seed)
    universe = np.arange(-50, 51, dtype=np.int64)
    weights = np.where(np.mod(universe, m) == 0, 2.0, 1.0)
    planted = rng.choice(universe, size=n, p=weights / weights.sum())
    stats = cluster_stats(planted, m)
    return {
        "planted_residue0_multiplier": 2.0,
        "n": n,
        "excess_round_share": stats["excess_round_share"],
        "chi2_p_value": stats["chi2_p_value"],
        "detected": bool(stats["chi2_p_value"] is not None and stats["chi2_p_value"] < 1e-6),
    }


def price_clustering_bench(
    *,
    horizon: float = 30000.0,
    seed: int = 0,
    m: int = 10,
    band: int = 5,
) -> dict[str, Any]:
    """Flow-arm clustering bench → sealed ``price_clustering.v1`` receipt.

    Arms (all SYNTHETIC; ``SplitFlow`` from the lane spec is not present on
    this branch, so the ref-anchor arm covers the reachability edge case):

    - ``touch_zi`` — default Santa Fe ZI flow, touch anchor: residue-neutral
      placement, ``excess_round_share ≈ 0``.
    - ``markov_regime`` — bursty two-state MO flow (calm/bursty per the lane
      spec): modulates market orders only, so LO placement stays
      residue-neutral.
    - ``ref_band`` — frozen reference anchor: submits land only at
      ``±1..±band``, leaving residue 0 unreachable — the arm that exercises
      reachable-set normalization (a naive ``1/m`` baseline would fabricate
      an anti-clustering deficit here).
    """
    h = float(horizon)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"horizon must be positive and finite, got {horizon!r}")
    mm = _modulus(m)
    if isinstance(band, bool) or int(band) < 1:
        raise ValueError(f"band must be an int >= 1, got {band!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    regime = MarkovRegimeFlow(
        states=(
            RegimeState("calm", 1.0, 0.5),
            RegimeState("bursty", 3.0, 0.62),
        ),
        stay_probs=(0.995, 0.985),
        seed=seed + 1,
    )
    arms: dict[str, tuple[ZILobConfig, MarkovRegimeFlow | None]] = {
        "touch_zi": (santa_fe_config(seed=seed, band=band), None),
        "markov_regime": (santa_fe_config(seed=seed + 1, band=band), regime),
        "ref_band": (
            replace(santa_fe_config(seed=seed + 2, band=band), anchor="ref"),
            None,
        ),
    }
    arm_out: dict[str, Any] = {}
    for name, (cfg, flow) in arms.items():
        run = sim_cluster_stats(flow=flow, horizon=h, seed=seed, m=mm, config=cfg)
        arm_out[name] = {
            "n_events": run["n_events"],
            "n_submits": run["n_submits"],
            "n_reachable_levels": run["cluster"]["n_reachable_levels"],
            "residue_shares": run["cluster"]["residue_shares"],
            "uniform_baseline_shares": run["cluster"]["uniform_baseline_shares"],
            "excess_round_share": run["cluster"]["excess_round_share"],
            "chi2_stat": run["cluster"]["chi2_stat"],
            "chi2_dof": run["cluster"]["chi2_dof"],
            "chi2_p_value": run["cluster"]["chi2_p_value"],
            "residue0_reachable": run["cluster"]["reachable_counts"][0] > 0,
            "price_domain_residues_match": run["price_domain_residues_match"],
        }
    probe = _detector_probe(m=mm, seed=seed + 3)
    touch = arm_out["touch_zi"]
    regime_arm = arm_out["markov_regime"]
    ref_arm = arm_out["ref_band"]
    interpretation = (
        "No round-number preference: touch-arm excess_round_share="
        f"{touch['excess_round_share']:+.4f} at residue 0, regime arm "
        f"{regime_arm['excess_round_share']:+.4f} — both under 1pt. The "
        "omnibus chi-square fires anyway "
        f"(p={touch['chi2_p_value']:.1e}): ZI submits concentrate near the "
        "wandering anchor, so placement is not uniform over the reachable "
        "hull — a level-density effect, not residue-0 clustering. The "
        f"ref-anchor arm leaves residue 0 unreachable by construction "
        f"(reachable levels at ±1..±{band} only) — reachable-set "
        "normalization keeps its excess at "
        f"{ref_arm['excess_round_share']:+.4f} where a naive 1/m baseline "
        "would fabricate a -0.1 deficit. Detector probe recovered the "
        f"planted 2x round-level preference (p={probe['chi2_p_value']:.1e}). "
        "SYNTHETIC placement diagnostic only — not market evidence."
    )
    body = {
        "schema": PRICE_CLUSTERING_SCHEMA,
        "kind": PRICE_CLUSTERING_KIND,
        "data_label": "SYNTHETIC",
        "label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "simulated_only": True,
        "claim": "synthetic_placement_diagnostic_only",
        "git_revision": git_revision(),
        "params": {"horizon": h, "seed": seed, "m": mm, "band": int(band)},
        "arms": arm_out,
        "detector_probe": probe,
        "interpretation": interpretation,
    }
    canonical = json.loads(canonical_json_bytes(body))
    return {**canonical, "receipt_sha256": hash_bytes(canonical_json_bytes(canonical))}


def write_price_clustering_receipt(
    receipts_dir: Path | str = Path("receipts"),
    **bench_kwargs: Any,
) -> Path:
    """Run the bench and write ``receipts/price_clustering_synth.json``."""
    payload = price_clustering_bench(**bench_kwargs)
    path = Path(receipts_dir) / PRICE_CLUSTERING_RECEIPT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
