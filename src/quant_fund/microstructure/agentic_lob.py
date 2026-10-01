"""Agentic-LOB phase-transition diagnostics on the zero-intelligence book.

**Labeled SYNTHETIC** research infrastructure (wave 18): Rosenzweig (2026,
arXiv:2609.31260) maps agent-populated limit order books onto open
statistical-mechanical systems and reports sharp first-order phase
transitions between three macroscopic regimes — a *collapsed/vaporized*
phase (insufficient liquidity provision: the book empties a side and never
recovers), a *continuous-liquidity* phase (orderly price discovery at finite
volatility), and a *frozen* phase (liquidity capacity compressed into too
little depth: mid-price volatility collapses toward zero). The paper's two
observables are mid-price volatility ``sigma`` measured in ticks and the
collapse probability ``p`` (empirical likelihood of a complete book
emptying). Its headline scaling laws are ``n_c = const`` (the collapse
boundary is invariant to depth) and ``d_c ∝ n`` (the frozen boundary scales
linearly with agent density).

This module implements the same diagnostic machinery on top of the repo's
existing event-driven ZI-LOB, where the thermodynamic control parameters map
onto order-flow intensities:

- liquidity-provision capacity ``n`` (paper) -> ``lam * band``: the per-side
  limit-order arrival rate on a ``band``-tick placement strip;
- observable market depth ``d`` (paper) -> the depth the (arrival, cancel)
  balance sustains, scanned via the per-order cancel rate ``theta_cxl`` and
  the placement strip ``band`` / ``density_exponent`` (order-size mix);
- driving temperature ``T`` (paper) -> the market-order intensity ``mu``
  and direction ``p_buy`` (drift-free flow at ``p_buy = 0.5``).

Phase boundaries are measured, not assumed: ``phase_diagram`` sweeps a
``(lam, theta_cxl)`` grid (optionally replicated over ``band`` values as the
order-size-mix axis), runs ``n_paths`` seeded book paths per cell, and
reports per-cell collapse rate, mid-price sigma in ticks, frozen rate,
resilience recovery times, and the coalesced phase label. Boundary
estimators extract the collapse threshold ``lam_c(theta)`` and fit the
frozen frontier — the analogues of the paper's two phase boundaries. A
documented deviation: in the paper ``n_c`` is invariant to ``d``, but the
ZI-LOB's collapse race is provision-vs-cancellation, so ``lam_c`` rises
with ``theta_cxl`` (roughly linearly at probe resolution) — ``lam_c_cv``
quantifies that slope honestly rather than asserting flatness.

``detect_phase_boundaries`` labels order-flow regime transitions inside one
path by segmenting the bucketed mid-volatility / OFI / depth channels with
``models.changepoint`` (lazy import — the sanctioned layer-order
cycle-breaker per docs/ARCHITECTURE_GUARDS.md; microstructure sits in
``market_data``, ``models`` in ``analytics``). ``phase_alarm`` composes
``metrics.watch`` (same layer): conformal p-values on a calibration bag of
volatility scores feed a composite jumper martingale, giving an
anytime-valid sequential alarm for entry into a high-volatility phase, plus
the Shiryaev-Roberts statistic. ``impact_zscore`` implements the paper's
Eq. (4) ensemble z-score between impacted and control mid-price paths and
classifies the dissipative / balanced / non-dissipative impact regimes of
Sec. 4. ``hysteresis_sweep`` quantifies initial-condition hysteresis
(thin-seeded vs thick-seeded books) — the first-order-transition signature.
``cluster_injection_phases`` is a documented supplement clustering per-path
observables into the three archetypes.

Composition — this module reuses, never reimplements:
- ``zi_lob_simulator.ZILobSimulator`` / ``ZILobConfig`` / ``santa_fe_config``
  — the matching engine, event clock, and seeded determinism;
- ``zi_lob_simulator.book_phase_metrics`` — per-sample spread/depth phase
  metrics composed inside the ensemble aggregate (single source of truth);
- ``models.changepoint`` (lazy) — BOCPD / binary segmentation / optimal
  partitioning for flow-regime transition boundaries;
- ``metrics.watch`` — ``CompositeJumper``, ``weighted_conformal_pvalue``,
  ``shiryaev_roberts`` for the sequential phase alarm;
- ``zi_lob_simulator.metaorder_impact_slope`` — owns the log-log sqrt-impact
  fit; ``impact_zscore_experiment`` here measures the *ensemble z-score
  propagation* (a different observable), so no duplication.

Honesty: every observable is a SYNTHETIC correctness diagnostic on a
zero-intelligence simulator, never market evidence; there is no broker
connectivity and no live-trading claim; no mark-to-market or PnL is produced
at all (the paper's diagnostics are state variables — volatility, collapse,
resilience — not returns). Fail-closed throughout: degenerate/short/invalid
inputs raise rather than silently degrade, and all randomness runs through
the simulator's seeded numpy Generator, so identical seeds give bit-identical
paths.

References:
- Rosenzweig, J. (2026). Agentic Limit Order Books: Phase Transitions and
  Market Impact. arXiv:2609.31260 [q-fin.TR]. Verified against
  https://arxiv.org/abs/2609.31260 and the full HTML text (fetched
  2026-09-30): three regimes (collapsed ``p > 90%``, continuous liquidity,
  frozen ``sigma << 1`` tick), scaling ``d_c ∝ n, n_c = const`` (Sec. 3.1,
  Eq. 3), ensemble impact z-score ``z(t) = (mu_I - mu_N)/sqrt(sigma_I
  sigma_N)`` (Sec. 4, Eq. 4), dissipative/balanced/non-dissipative impact
  regimes at ``sigma ~ 4.21`` ticks (Sec. 4.1), cracking/melting in the
  frozen phase at ``sigma ~ 0.1`` ticks (Sec. 4.2).
- Cont, Stoikov, Talreja (2010). A stochastic model for order book dynamics.
  *Operations Research* 58(3):549-563 — the ZI queueing engine underneath.
- Adams, MacKay (2007). Bayesian online changepoint detection.
  arXiv:0710.3742 — composed via ``models.changepoint.bocpd_gaussian``.
- Vovk (2021). Testing randomness online. arXiv:2105.08669 — the simple
  jumper martingale behind the sequential alarm (via ``metrics.watch``).
- Farmer, Foley (2009). The economy needs agent-based modelling. *Nature*
  460:685-686 — the paper's framing motivation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.watch import (
    CompositeJumper,
    shiryaev_roberts,
    weighted_conformal_pvalue,
)
from quant_fund.microstructure.zi_lob_simulator import (
    BookSample,
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
    book_phase_metrics,
    santa_fe_config,
)
from quant_fund.utils.series import finite_series

Array = NDArray[np.float64]
IdxArray = NDArray[np.intp]
BoolArray = NDArray[np.bool_]

AGENTIC_LOB_REVISION = "SYNTHETIC_AGENTIC_LOB_v1"

#: The three macroscopic phases of Rosenzweig (2026) Table 1, in the
#: canonical order used by the deterministic cluster archetypes.
PHASE_LABELS: tuple[str, str, str] = ("collapsed", "continuous_liquidity", "frozen")

#: Impact-propagation regimes of Rosenzweig (2026) Sec. 4.1-4.2.
IMPACT_REGIMES: tuple[str, ...] = ("dissipative", "balanced", "non_dissipative", "melting")

#: Boundary-detection methods, mapped onto ``models.changepoint`` detectors.
BOUNDARY_METHODS: tuple[str, str, str] = ("optimal", "binseg", "bocpd")

__all__ = [
    "AGENTIC_LOB_REVISION",
    "BOUNDARY_METHODS",
    "IMPACT_REGIMES",
    "PHASE_LABELS",
    "PhaseScanConfig",
    "bench_agentic_lob",
    "classify_phase",
    "cluster_injection_phases",
    "detect_phase_boundaries",
    "ensemble_phase",
    "hysteresis_sweep",
    "impact_zscore",
    "impact_zscore_experiment",
    "phase_alarm",
    "phase_diagram",
    "phase_feature_series",
    "run_phase_path",
]


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (house style; mirrors zi_lob_simulator)
# ---------------------------------------------------------------------------


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


def _prob(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {x!r}")
    return v


def _int_at_least(x: int, lo: int, name: str) -> int:
    if isinstance(x, bool) or int(x) < lo:
        raise ValueError(f"{name} must be an int >= {lo}, got {x!r}")
    return int(x)


def _seq_of_pos(values: Sequence[float], name: str) -> list[float]:
    out = [_pos_finite(v, f"{name}[{i}]") for i, v in enumerate(values)]
    if not out:
        raise ValueError(f"{name} must be non-empty")
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing, got {list(values)!r}")
    return out


# ---------------------------------------------------------------------------
# Scan configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseScanConfig:
    """Protocol constants for one phase-scan cell / path ensemble.

    ``horizon`` is the recorded window in simulated seconds after a
    ``warmup`` burn-in. ``sample_interval`` is the fixed recording grid
    (mid/spread/depth sampled on it). Collapse bookkeeping runs at event
    resolution, not the sample grid: an *empty-side episode* opens the event
    a side's depth hits zero and closes when that side's depth first reaches
    ``recovery_depth`` orders again — the resilience recovery time. An
    episode lasting ``collapse_min_seconds`` or longer marks the path
    *collapsed* (the paper's "complete emptying before path completion"
    generalized to a dwell-time criterion: transient re-seeds at rate
    ``lam * band`` are not collapses). ``frozen_sigma_step`` is the
    per-interval mid-volatility threshold below which a non-collapsed path
    is *frozen* (paper: ``sigma << 1`` tick). ``collapse_rate_threshold`` is
    the ensemble fraction of collapsed paths at which a cell is labeled
    ``collapsed`` (paper: ``p > 90%``; default 0.5 keeps the boundary at the
    median-path transition).
    """

    horizon: float = 300.0
    warmup: float = 150.0
    sample_interval: float = 1.0
    n_paths: int = 6
    collapse_min_seconds: float = 8.0
    recovery_depth: int = 2
    frozen_sigma_step: float = 0.05
    collapse_rate_threshold: float = 0.5
    min_defined_mids: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        _pos_finite(self.horizon, "horizon")
        _pos_finite(self.warmup, "warmup")
        _pos_finite(self.sample_interval, "sample_interval")
        _pos_finite(self.collapse_min_seconds, "collapse_min_seconds")
        _pos_finite(self.frozen_sigma_step, "frozen_sigma_step")
        _prob(self.collapse_rate_threshold, "collapse_rate_threshold")
        _int_at_least(self.n_paths, 1, "n_paths")
        _int_at_least(self.recovery_depth, 1, "recovery_depth")
        _int_at_least(self.min_defined_mids, 2, "min_defined_mids")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")


# ---------------------------------------------------------------------------
# Per-path simulation: observables on the paper's two order parameters
# ---------------------------------------------------------------------------


class _EmptySideTracker:
    """Per-side empty-depth episodes at event resolution (resilience clock).

    An episode opens when a side's depth hits 0 and closes when that side's
    depth reaches ``recovery_depth``; episodes still open at path end are
    recorded as unrecovered. Durations feed the resilience diagnostics.
    """

    def __init__(self, recovery_depth: int) -> None:
        self.recovery_depth = int(recovery_depth)
        self._open: dict[str, float] = {}
        # (start_time, duration) pairs: recovered episodes and ones still
        # open at path end (bookkept separately so an unrecovered collapse
        # is never silently scored as a recovery).
        self.durations: list[tuple[float, float]] = []
        self.unrecovered: list[tuple[float, float]] = []

    def observe(self, t: float, bid_depth: int, ask_depth: int) -> None:
        for side, depth in (("buy", bid_depth), ("sell", ask_depth)):
            if side not in self._open and depth == 0:
                self._open[side] = t
            elif side in self._open and depth >= self.recovery_depth:
                t0 = self._open.pop(side)
                self.durations.append((t0, t - t0))

    def close(self, t_end: float) -> None:
        for t0 in self._open.values():
            self.unrecovered.append((t0, t_end - t0))
        self._open.clear()

    @property
    def n_episodes(self) -> int:
        return len(self.durations) + len(self.unrecovered)


def _mid_in_ticks(sim: ZILobSimulator) -> float:
    """Mid price in tick units relative to ``s0`` (nan when one-sided)."""
    bb, ba = sim.best_bid_level, sim.best_ask_level
    if bb is None or ba is None:
        return float("nan")
    return 0.5 * float(bb + ba)


def run_phase_path(
    config: ZILobConfig,
    scan: PhaseScanConfig | None = None,
    *,
    flow: MarkovRegimeFlow | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate one ZI-LOB path and measure the paper's phase observables.

    Warms up ``scan.warmup`` seconds, then records (mid in ticks, spread,
    per-side depth) on the fixed ``sample_interval`` grid for ``horizon``
    seconds while tracking empty-side episodes at event resolution. Reports:

    - ``sigma_ticks``: std of the recorded mid trajectory in ticks — the
      paper's ``sigma`` (trajectory dispersion; horizon-dependent);
    - ``sigma_step_ticks``: std of consecutive mid increments in ticks —
      the horizon-robust volatility used for the frozen boundary;
    - ``collapsed`` / ``collapse_time`` / ``n_empty_episodes`` /
      ``empty_time_frac`` / ``max_empty_seconds``: collapse bookkeeping
      (an episode >= ``collapse_min_seconds`` marks the path collapsed);
    - ``resilience_mean_seconds`` / ``resilience_max_seconds`` /
      ``n_unrecovered_episodes``: recovery-time diagnostics (the time an
      emptied side needs to regain ``recovery_depth`` orders — diverges
      approaching the collapse boundary, the critical-slowing signature);
    - ``mean_total_depth``, ``mean_spread_ticks``, ``mid_defined_frac``,
      ``n_trades``, ``buy_fraction``.

    Deterministic: path ``i`` of an ensemble should be drawn with
    ``seed=scan.seed + i`` (done by ``ensemble_phase``); identical seeds give
    bit-identical paths.
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    cfg = config if seed is None else replace(config, seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    sim.run(sc.warmup)

    tracker = _EmptySideTracker(sc.recovery_depth)
    tracker.observe(sim.t, sim.bid_depth, sim.ask_depth)

    mids: list[float] = []
    spreads: list[float] = []
    depths: list[float] = []
    samples: list[BookSample] = [sim.sample()]
    t_end = sc.warmup + sc.horizon
    t_next = sim.t + sc.sample_interval
    while sim.t < t_end:
        sim.step()
        tracker.observe(sim.t, sim.bid_depth, sim.ask_depth)
        if sim.t >= t_next:
            samples.append(sim.sample())
            mids.append(_mid_in_ticks(sim))
            sp = sim.spread_ticks
            spreads.append(float(sp) if sp is not None else float("nan"))
            depths.append(float(sim.total_depth))
            while t_next <= sim.t:
                t_next += sc.sample_interval
    tracker.close(t_end)

    mid = np.asarray(mids, dtype=np.float64)
    finite_mid = mid[np.isfinite(mid)]
    n_def = int(finite_mid.size)
    if n_def < sc.min_defined_mids:
        # A path whose mid is almost never defined is deep in the collapsed
        # phase; its volatility is undefined and honestly flagged, not zeroed.
        sigma_ticks = float("nan")
        sigma_step = float("nan")
    else:
        sigma_ticks = float(finite_mid.std(ddof=1))
        # Consecutive-increment vol, only across adjacent *defined* samples.
        step_diffs: list[float] = []
        defined_idx = np.flatnonzero(np.isfinite(mid))
        for a, b in zip(defined_idx[:-1], defined_idx[1:], strict=False):
            if b - a == 1:
                step_diffs.append(mid[b] - mid[a])
        sd = np.asarray(step_diffs, dtype=np.float64)
        sigma_step = float(sd.std(ddof=1)) if sd.size >= 2 else float("nan")
    eps = tracker.durations + tracker.unrecovered
    durations = [d for _t0, d in eps]
    max_empty = float(max(durations)) if durations else 0.0
    empty_time = float(sum(durations))
    collapsed = bool(max_empty >= sc.collapse_min_seconds)
    collapse_time = min(t0 for t0, d in eps if d >= sc.collapse_min_seconds) if collapsed else None
    signs = [1.0 if tr.aggressor == "buy" else -1.0 for tr in sim.trades]
    n_tr = len(signs)
    recovery_times = [d for _t0, d in tracker.durations]
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "seed": cfg.seed,
        "sigma_ticks": sigma_ticks,
        "sigma_step_ticks": sigma_step,
        "collapsed": collapsed,
        "collapse_time": collapse_time,
        "n_empty_episodes": tracker.n_episodes,
        "n_unrecovered_episodes": len(tracker.unrecovered),
        "empty_time_frac": empty_time / sc.horizon,
        "max_empty_seconds": max_empty,
        "resilience_mean_seconds": float(np.mean(recovery_times))
        if recovery_times
        else float("nan"),
        "resilience_max_seconds": max_empty,
        "mean_total_depth": float(np.mean(depths)) if depths else float("nan"),
        "mean_spread_ticks": float(np.nanmean(np.asarray(spreads)))
        if spreads and np.isfinite(np.asarray(spreads)).any()
        else float("nan"),
        "mid_defined_frac": n_def / max(len(mids), 1),
        "n_trades": n_tr,
        "buy_fraction": float(np.mean([s > 0.0 for s in signs])) if n_tr else float("nan"),
        "n_samples": len(mids),
        "n_events": sim.n_events,
        # Spread/depth phase metrics of the recorded samples, composed from
        # zi_lob_simulator.book_phase_metrics (single source of truth).
        "book_phase": book_phase_metrics(samples),
        "event_counts": sim.event_counts(),
    }


def classify_phase(
    sigma_step_ticks: float,
    collapse_rate: float,
    scan: PhaseScanConfig | None = None,
) -> str:
    """Deterministic three-phase label for one cell (paper Table 1).

    ``collapsed`` when the ensemble collapse rate reaches
    ``scan.collapse_rate_threshold``; ``frozen`` when the surviving mid-price
    per-interval volatility is below ``scan.frozen_sigma_step`` (the paper's
    ``sigma << 1`` tick criterion); ``continuous_liquidity`` otherwise. A
    cell with no defined volatility (every path collapsed) is ``collapsed``
    by exhaustion — a nan sigma with a sub-threshold collapse rate raises
    rather than guessing.
    """
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    cr = _prob(collapse_rate, "collapse_rate")
    s = float(sigma_step_ticks)
    if not math.isfinite(s):
        if cr >= sc.collapse_rate_threshold or cr > 0.0:
            return "collapsed"
        raise ValueError("sigma_step_ticks must be finite for a non-collapsed cell")
    if cr >= sc.collapse_rate_threshold:
        return "collapsed"
    if s < sc.frozen_sigma_step:
        return "frozen"
    return "continuous_liquidity"


def ensemble_phase(
    config: ZILobConfig,
    scan: PhaseScanConfig | None = None,
    *,
    n_paths: int | None = None,
) -> dict[str, Any]:
    """Monte-Carlo cell: ``n_paths`` seeded paths -> phase observables.

    Path ``i`` draws seed ``config.seed + i`` (bit-identical reruns). The
    cell label applies ``classify_phase`` to the median per-interval sigma
    over non-collapsed paths and the collapse rate. ``book_phase_metrics``
    is composed over the aggregated sample statistics rather than
    reimplemented. ``resilience_*`` aggregates the per-path recovery means.
    """
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    k = sc.n_paths if n_paths is None else _int_at_least(n_paths, 1, "n_paths")
    paths = [run_phase_path(config, sc, seed=config.seed + i) for i in range(k)]
    collapses = np.asarray([bool(p["collapsed"]) for p in paths], dtype=np.float64)
    collapse_rate = float(collapses.mean())
    sig_step = np.asarray([float(p["sigma_step_ticks"]) for p in paths], dtype=np.float64)
    sig_traj = np.asarray([float(p["sigma_ticks"]) for p in paths], dtype=np.float64)
    surv = sig_step[np.isfinite(sig_step)]
    sigma_med = float(np.median(surv)) if surv.size else float("nan")
    phase = classify_phase(sigma_med, collapse_rate, sc)
    res = [float(p["resilience_mean_seconds"]) for p in paths]
    res_f = np.asarray([r for r in res if math.isfinite(r)], dtype=np.float64)
    frozen_rate = float(
        np.mean((sig_step < sc.frozen_sigma_step) & np.isfinite(sig_step) & (collapses == 0.0))
    )
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "lam": config.lam,
        "theta_cxl": config.theta_cxl,
        "mu": config.mu,
        "band": config.band,
        "density_exponent": config.density_exponent,
        "p_buy": config.p_buy,
        "phase": phase,
        "collapse_rate": collapse_rate,
        "frozen_rate": frozen_rate,
        "sigma_step_median_ticks": sigma_med,
        "sigma_step_mean_ticks": float(np.mean(surv)) if surv.size else float("nan"),
        "sigma_ticks_median": float(np.nanmedian(sig_traj))
        if np.isfinite(sig_traj).any()
        else float("nan"),
        "resilience_mean_seconds": float(np.mean(res_f)) if res_f.size else float("nan"),
        "resilience_max_seconds": float(max(float(p["resilience_max_seconds"]) for p in paths)),
        "mean_total_depth": float(np.mean([float(p["mean_total_depth"]) for p in paths])),
        "mean_spread_ticks": float(np.nanmean([float(p["mean_spread_ticks"]) for p in paths]))
        if any(math.isfinite(float(p["mean_spread_ticks"])) for p in paths)
        else float("nan"),
        "mid_defined_frac": float(np.mean([float(p["mid_defined_frac"]) for p in paths])),
        "n_paths": k,
        "n_paths_defined_sigma": int(surv.size),
        "sigma_step_per_path": sig_step.tolist(),
        "collapse_per_path": collapses.astype(bool).tolist(),
    }


# ---------------------------------------------------------------------------
# Phase diagram: (lam, theta) grid with boundary estimation
# ---------------------------------------------------------------------------


def _collapse_boundary(lams: list[float], collapse_rates: Array, threshold: float) -> float:
    """Interior collapse boundary in ``lam`` (nan when outside the grid).

    The smallest ``lam`` whose collapse rate is below ``threshold``, but only
    when a genuine crossing is bracketed: if the lowest grid point is already
    below threshold the boundary lies off-grid below (censored, nan), and if
    no point clears the threshold the whole column is collapsed (nan).
    """
    crs = [float(c) for c in collapse_rates]
    if crs[0] < threshold:
        return float("nan")
    for lam, cr in zip(lams, crs, strict=False):
        if cr < threshold:
            return float(lam)
    return float("nan")


def phase_diagram(
    *,
    base_config: ZILobConfig,
    lam_values: Sequence[float],
    theta_values: Sequence[float],
    scan: PhaseScanConfig | None = None,
    band_values: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Phase diagram over a ``(lam, theta_cxl)`` grid (optionally x ``band``).

    For every grid cell runs ``ensemble_phase`` — the paper's Monte-Carlo
    observables (``sigma`` in ticks, collapse probability ``p``) with the
    ZI-LOB control mapping ``n -> lam * band`` and depth scanned through the
    cancel rate. ``band_values`` replicates the grid per placement-strip
    width, the order-size-mix axis of the lane.

    Boundary estimation (paper Sec. 3.1, Eq. 3):
    - ``lam_c_by_theta[j]`` = smallest ``lam`` with collapse rate below the
      threshold at fixed ``theta_values[j]``; ``lam_c_cv`` is its
      coefficient of variation across theta — the paper's ``n_c = const``
      invariance check (small cv = flat boundary).
    - ``frozen_boundary_*``: per ``theta``, the interior ``lam`` crossing
      where the median per-interval sigma drops below ``frozen_sigma_step``
      as the book deepens (the continuous->frozen frontier, bracketed
      crossings only); a least-squares ``theta ~ a*lam + b`` fit on those
      boundary points is the analogue of the paper's ``d_c ∝ n`` scaling.
    - ``order_parameter_monotone``: whether median sigma is non-decreasing in
      ``theta`` at fixed ``lam`` (cancel pressure deepens disorder), a
      coarse sanity check on the measured phase structure.
    """
    if not isinstance(base_config, ZILobConfig):
        raise TypeError("base_config must be a ZILobConfig")
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    lams = _seq_of_pos(lam_values, "lam_values")
    thetas = _seq_of_pos(theta_values, "theta_values")
    bands = [int(base_config.band)] if band_values is None else [int(b) for b in band_values]
    if band_values is not None:
        for b in band_values:
            if isinstance(b, bool) or int(b) < 1 or float(b) != int(b):
                raise ValueError(f"band_values must be ints >= 1, got {band_values!r}")
    if not bands:
        raise ValueError("band_values must be non-empty when given")

    per_band: list[dict[str, Any]] = []
    for band in bands:
        cells: list[list[dict[str, Any]]] = []
        for th in thetas:
            row: list[dict[str, Any]] = []
            for la in lams:
                cfg = replace(base_config, lam=la, theta_cxl=th, band=band)
                row.append(ensemble_phase(cfg, sc))
            cells.append(row)
        sigma = np.asarray(
            [[c["sigma_step_median_ticks"] for c in row] for row in cells], dtype=np.float64
        )
        collapse = np.asarray(
            [[c["collapse_rate"] for c in row] for row in cells], dtype=np.float64
        )
        frozen = np.asarray([[c["frozen_rate"] for c in row] for row in cells], dtype=np.float64)
        phase_grid = [[str(c["phase"]) for c in row] for row in cells]
        lam_c = np.asarray(
            [
                _collapse_boundary(lams, collapse[j], sc.collapse_rate_threshold)
                for j in range(len(thetas))
            ],
            dtype=np.float64,
        )
        lam_c_finite = lam_c[np.isfinite(lam_c)]
        lam_c_cv = (
            float(lam_c_finite.std() / lam_c_finite.mean())
            if lam_c_finite.size >= 2 and float(lam_c_finite.mean()) > 0.0
            else float("nan")
        )
        # Frozen frontier: per theta, the interior lam crossing where the
        # median per-interval sigma drops below frozen_sigma_step as the
        # book deepens (the frozen<-continuous boundary, analogue of d_c(n)).
        # Only bracketed crossings count — a column frozen at the lowest
        # grid point is censored off-grid and contributes nothing.
        fb_lam: list[float] = []
        fb_theta: list[float] = []
        for j, th in enumerate(thetas):
            col = sigma[j]
            for i in range(len(lams) - 1):
                a_s, b_s = float(col[i]), float(col[i + 1])
                if not (math.isfinite(a_s) and math.isfinite(b_s)):
                    continue
                if a_s >= sc.frozen_sigma_step > b_s:
                    fb_lam.append(0.5 * (lams[i] + lams[i + 1]))
                    fb_theta.append(th)
                    break
        if len(fb_lam) >= 2 and np.std(fb_lam) > 0.0 and np.std(fb_theta) > 0.0:
            a, b = np.polyfit(np.asarray(fb_lam), np.asarray(fb_theta), 1)
            frozen_slope, frozen_intercept = float(a), float(b)
            fb_r = float(np.corrcoef(np.asarray(fb_lam), np.asarray(fb_theta))[0, 1])
        else:
            frozen_slope = frozen_intercept = fb_r = float("nan")
        # Order-parameter monotonicity: sigma non-decreasing in theta per lam.
        mono_cols = 0
        mono_ok = 0
        for i in range(len(lams)):
            col = sigma[:, i]
            col = col[np.isfinite(col)]
            if col.size >= 2:
                mono_cols += 1
                if bool(np.all(np.diff(col) >= -1e-12)):
                    mono_ok += 1
        per_band.append(
            {
                "band": band,
                "cells": cells,
                "sigma_step_grid": sigma.tolist(),
                "collapse_grid": collapse.tolist(),
                "frozen_grid": frozen.tolist(),
                "phase_grid": phase_grid,
                "lam_c_by_theta": lam_c.tolist(),
                "lam_c_cv": lam_c_cv,
                "frozen_boundary_lam": fb_lam,
                "frozen_boundary_theta": fb_theta,
                "frozen_boundary_slope": frozen_slope,
                "frozen_boundary_intercept": frozen_intercept,
                "frozen_boundary_corr": fb_r,
                "n_theta_cols_monotone_sigma": mono_cols,
                "n_theta_cols_monotone_ok": mono_ok,
            }
        )
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "lam_values": lams,
        "theta_values": thetas,
        "per_band": per_band,
        "n_bands": len(bands),
        "n_cells_total": len(bands) * len(lams) * len(thetas),
        "horizon": sc.horizon,
        "n_paths": sc.n_paths,
        "collapse_rate_threshold": sc.collapse_rate_threshold,
        "frozen_sigma_step": sc.frozen_sigma_step,
        "claim": "simulator_internal_diagnostic_only",
    }


# ---------------------------------------------------------------------------
# Bucketed feature series + changepoint boundary detection + sequential alarm
# ---------------------------------------------------------------------------


def phase_feature_series(
    config: ZILobConfig,
    scan: PhaseScanConfig | None = None,
    *,
    bucket_seconds: float = 10.0,
    flow: MarkovRegimeFlow | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bucket one path into (mid-volatility, OFI, depth, empty-frac) channels.

    Buckets of ``bucket_seconds`` simulated seconds on the post-warmup
    window; per bucket: ``mid_sigma_ticks`` (std of sampled mids inside the
    bucket — the local activity level), ``ofi`` (signed trade imbalance
    ``(n_buy - n_sell) / n_trades`` in the bucket), ``mean_total_depth``,
    ``empty_frac`` (fraction of bucket time with a side empty), and
    ``mean_spread_ticks``. These channels feed ``detect_phase_boundaries``
    and ``phase_alarm``; a ``flow`` (e.g. ``MarkovRegimeFlow``) injects
    order-flow regime transitions so boundaries are *planted*, keeping the
    detection an honest correctness test.
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    bsec = _pos_finite(bucket_seconds, "bucket_seconds")
    cfg = config if seed is None else replace(config, seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    sim.run(sc.warmup)
    t0 = sim.t
    n_buckets = int(sc.horizon / bsec)
    if n_buckets < 2:
        raise ValueError(f"horizon {sc.horizon} too short for >= 2 buckets of {bsec}s")
    # Per-bucket event-resolution integration of empty-side dwell (each
    # inter-event interval is attributed to the bucket it opened in).
    bucket_mid: list[list[float]] = [[] for _ in range(n_buckets)]
    bucket_depth: list[list[float]] = [[] for _ in range(n_buckets)]
    bucket_spread: list[list[float]] = [[] for _ in range(n_buckets)]
    empty_dwell = np.zeros(n_buckets, dtype=np.float64)
    prev_t = sim.t
    prev_empty = sim.bid_depth == 0 or sim.ask_depth == 0
    trade_cursor = 0
    t_next = sim.t + sc.sample_interval
    t_end = t0 + sc.horizon
    while sim.t < t_end:
        sim.step()
        dt = sim.t - prev_t
        if prev_empty:
            b_idx = min(int((prev_t - t0) / bsec), n_buckets - 1)
            if b_idx >= 0:
                empty_dwell[b_idx] += dt
        prev_t = sim.t
        prev_empty = sim.bid_depth == 0 or sim.ask_depth == 0
        if sim.t >= t_next:
            b_idx = int((sim.t - t0) / bsec)
            if 0 <= b_idx < n_buckets:
                bucket_mid[b_idx].append(_mid_in_ticks(sim))
                bucket_depth[b_idx].append(float(sim.total_depth))
                sp = sim.spread_ticks
                bucket_spread[b_idx].append(float(sp) if sp is not None else float("nan"))
            while t_next <= sim.t:
                t_next += sc.sample_interval
    mid_sigma = np.zeros(n_buckets)
    ofi = np.zeros(n_buckets)
    depth_mean = np.zeros(n_buckets)
    spread_mean = np.zeros(n_buckets)
    n_mo = np.zeros(n_buckets)
    for b in range(n_buckets):
        m = np.asarray(bucket_mid[b], dtype=np.float64)
        m = m[np.isfinite(m)]
        mid_sigma[b] = float(m.std(ddof=1)) if m.size >= 2 else float("nan")
        depth_mean[b] = float(np.mean(bucket_depth[b])) if bucket_depth[b] else float("nan")
        spr = np.asarray(bucket_spread[b], dtype=np.float64)
        spread_mean[b] = float(np.nanmean(spr)) if spr.size else float("nan")
        b_lo, b_hi = t0 + b * bsec, t0 + (b + 1) * bsec
        buys = sells = 0
        while trade_cursor < len(sim.trades) and sim.trades[trade_cursor].t < b_hi:
            tr = sim.trades[trade_cursor]
            trade_cursor += 1
            if tr.t >= b_lo:
                if tr.aggressor == "buy":
                    buys += 1
                else:
                    sells += 1
        tot = buys + sells
        n_mo[b] = float(tot)
        ofi[b] = (buys - sells) / tot if tot else float("nan")
    centers = t0 + (np.arange(n_buckets) + 0.5) * bsec
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "t_bucket_center": centers,
        "bucket_seconds": bsec,
        "mid_sigma_ticks": mid_sigma,
        "ofi": ofi,
        "mean_total_depth": depth_mean,
        "empty_frac": empty_dwell / bsec,
        "mean_spread_ticks": spread_mean,
        "n_mo_per_bucket": n_mo,
        "n_buckets": n_buckets,
    }


def _channel_boundaries(series: Array, method: str) -> IdxArray:
    """Dispatch a scalar channel to the composed changepoint detector (lazy)."""
    from quant_fund.models.changepoint import (
        binary_segmentation,
        bocpd_gaussian,
        optimal_partition_mean,
    )

    if method == "optimal":
        cps, _obj = optimal_partition_mean(series)
        return np.asarray(cps, dtype=np.intp)
    if method == "binseg":
        return np.asarray(binary_segmentation(series), dtype=np.intp)
    if method == "bocpd":
        out = bocpd_gaussian(series)
        return np.asarray(out["changepoints"], dtype=np.intp)
    raise ValueError(f"method must be one of {BOUNDARY_METHODS}, got {method!r}")


def detect_phase_boundaries(
    features: Mapping[str, Array] | Array,
    *,
    method: str = "optimal",
    scan: PhaseScanConfig | None = None,
    channels: Sequence[str] = ("mid_sigma_ticks", "ofi"),
    min_segment: int = 2,
) -> dict[str, Any]:
    """Coalesced phase labels per segment of a bucketed feature series.

    Runs ``models.changepoint`` detection on each channel (default the
    mid-volatility and order-flow-imbalance channels — nan entries are
    median-imputed per channel so sparse buckets do not break detection),
    unions the boundaries, and labels each segment by its mean activity:
    ``collapsed`` when mean ``empty_frac`` (if present) is at least
    ``scan.collapse_rate_threshold``, ``frozen`` when mean
    ``mid_sigma_ticks`` is below ``scan.frozen_sigma_step``, else
    ``continuous_liquidity``. Identical adjacent labels coalesce into
    macro-phase segments — the paper's regime-shift machinery applied on
    the intra-path clock. Composes ``models.changepoint`` lazily (layer
    boundary per docs/ARCHITECTURE_GUARDS.md).
    """
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    if method not in BOUNDARY_METHODS:
        raise ValueError(f"method must be one of {BOUNDARY_METHODS}, got {method!r}")
    ms = _int_at_least(min_segment, 1, "min_segment")
    if isinstance(features, Mapping):
        data: dict[str, Array] = {}
        for k, v in features.items():
            try:
                arr = np.asarray(v, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                continue  # non-numeric metadata (label strings, revisions)
            if arr.size <= 1:
                continue  # scalar metadata (seed, n_buckets), not a channel
            data[str(k)] = arr
    else:
        data = {"mid_sigma_ticks": np.asarray(features, dtype=np.float64).reshape(-1)}
    if not data:
        raise ValueError("features must contain at least one channel")
    n = len(next(iter(data.values())))
    if any(v.shape[0] != n for v in data.values()):
        raise ValueError("all feature channels must share length")
    if n < 4:
        raise ValueError(f"need at least 4 buckets for boundary detection, got {n}")
    used = [c for c in channels if c in data]
    if not used:
        raise ValueError(f"none of the channels {list(channels)} present in features")
    bounds: set[int] = set()
    for ch in used:
        s = data[ch].copy()
        finite = np.isfinite(s)
        if not finite.any():
            continue
        med = float(np.median(s[finite]))
        s[~finite] = med
        if float(s.std()) <= 0.0:
            continue
        for cp in _channel_boundaries(s, method):
            if 0 < int(cp) < n:
                bounds.add(int(cp))
    edges = sorted(bounds)
    seg_edges = [0, *edges, n]
    segments: list[dict[str, Any]] = []
    for a, b in zip(seg_edges[:-1], seg_edges[1:], strict=False):
        if b - a < ms and segments:
            # Sub-minimum slivers merge into the previous segment.
            segments[-1]["end"] = b
            continue
        sigma_mean = (
            float(np.nanmean(data["mid_sigma_ticks"][a:b]))
            if "mid_sigma_ticks" in data
            else float("nan")
        )
        empty_mean = float(np.nanmean(data["empty_frac"][a:b])) if "empty_frac" in data else 0.0
        ofi_mean = float(np.nanmean(data["ofi"][a:b])) if "ofi" in data else float("nan")
        label = classify_phase(sigma_mean, empty_mean, sc)
        segments.append(
            {
                "start": a,
                "end": b,
                "phase": label,
                "mean_sigma_step": sigma_mean,
                "mean_empty_frac": empty_mean,
                "mean_ofi": ofi_mean,
            }
        )
    # Coalesce adjacent identical labels into macro-phase segments.
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and merged[-1]["phase"] == seg["phase"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "method": method,
        "boundaries": edges,
        "segments": merged,
        "n_segments": len(merged),
        "n_buckets": n,
        "channels_used": used,
    }


def phase_alarm(
    scores: Array,
    *,
    calibration_size: int | None = None,
    alpha: float = 0.05,
    jump_rates: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0),
    seed: int = 0,
) -> dict[str, Any]:
    """Anytime-valid sequential alarm for entry into a volatile phase.

    Composes ``metrics.watch``: the first ``calibration_size`` scores form a
    fixed calibration bag; each subsequent score gets a randomized conformal
    p-value via ``weighted_conformal_pvalue`` (seeded uniform tie-breaker),
    which updates a ``CompositeJumper`` betting martingale. Under an
    exchangeable continuation of the calm phase the wealth path is a test
    martingale and Ville bounds P(ever >= 1/alpha) <= alpha — a sequential
    phase-change alarm with no peeking penalty. Post-calibration the bag is
    frozen, matching the WATCHMonitor convention. Returns the wealth path,
    first crossing index (on the test axis), and the Shiryaev-Roberts
    statistic of the terminal wealth path.
    """
    x = finite_series(np.asarray(scores, dtype=np.float64), 4)
    a = _prob(alpha, "alpha")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    cal = (
        x.size // 4
        if calibration_size is None
        else _int_at_least(calibration_size, 2, "calibration_size")
    )
    if cal >= x.size:
        raise ValueError(f"calibration_size {cal} must leave at least one test score (n={x.size})")
    bag = x[:cal].copy()
    jumper = CompositeJumper(jump_rates)
    rng = np.random.default_rng(seed)
    wealth = np.empty(x.size - cal, dtype=np.float64)
    alarm_level = 1.0 / a
    first_cross: int | None = None
    for i, s in enumerate(x[cal:]):
        p = weighted_conformal_pvalue(
            np.concatenate([bag, np.asarray([s])]),
            test_index=-1,
            tie_breaker=float(rng.uniform()),
        )
        wealth[i] = jumper.update(p)
        if first_cross is None and wealth[i] >= alarm_level:
            first_cross = i
    path_with_prior = np.concatenate([[1.0], wealth])
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "wealth": wealth,
        "wealth_final": float(wealth[-1]),
        "alarm": first_cross is not None,
        "first_cross": first_cross,
        "alarm_level": float(alarm_level),
        "shiryaev_roberts_final": float(shiryaev_roberts(path_with_prior)),
        "calibration_size": cal,
        "n_test": int(x.size - cal),
        "alpha": a,
    }


# ---------------------------------------------------------------------------
# Ensemble impact z-score (paper Sec. 4, Eq. 4) + regime classification
# ---------------------------------------------------------------------------


def impact_zscore(impacted: Array, control: Array, *, min_defined: int = 2) -> dict[str, Any]:
    """Paper Eq. (4): ``z(t) = (mu_I(t) - mu_N(t)) / sqrt(sigma_I(t) sigma_N(t))``.

    ``impacted`` / ``control`` are ``(n_paths, n_times)`` ensembles of aligned
    mid traces (in ticks). ``sigma`` here is the cross-path ensemble standard
    deviation at time ``t``. Per time, the statistic uses only paths whose
    mid is defined (a side-empty path has no mid and is honestly skipped);
    times with fewer than 2 defined paths or zero variance on either
    ensemble leave ``z`` undefined (nan, counted via ``n_defined`` — the
    honest counterpart of the paper's identical-initial-condition warm
    start). Fewer than ``min_defined`` usable times raises. Entries that are
    +/-inf raise; nan is the supported missing-quote marker.
    """
    imp = np.asarray(impacted, dtype=np.float64)
    con = np.asarray(control, dtype=np.float64)
    if imp.ndim != 2 or con.ndim != 2:
        raise ValueError("impacted and control must be (n_paths, n_times) arrays")
    if imp.shape[1] != con.shape[1]:
        raise ValueError("impacted and control must share the time axis")
    if imp.shape[0] < 2 or con.shape[0] < 2:
        raise ValueError("need at least 2 paths per ensemble")
    if imp.shape[1] < 2:
        raise ValueError("need at least 2 times")
    if np.isinf(imp).any() or np.isinf(con).any():
        raise ValueError("ensembles must not contain infinities")

    def _stats(arr: Array) -> tuple[Array, Array, BoolArray]:
        fin = np.isfinite(arr)
        n = fin.sum(axis=0)
        ok = n >= 2
        s = np.where(fin, arr, 0.0)
        mu = np.full(arr.shape[1], np.nan, dtype=np.float64)
        sd = np.full(arr.shape[1], np.nan, dtype=np.float64)
        if ok.any():
            mu[ok] = s.sum(axis=0)[ok] / n[ok]
            dev2 = np.where(fin, (arr - mu) ** 2, 0.0)
            sd[ok] = np.sqrt(dev2.sum(axis=0)[ok] / (n[ok] - 1))
        return mu, sd, ok

    mu_i, sd_i, ok_i = _stats(imp)
    mu_n, sd_n, ok_n = _stats(con)
    denom = np.sqrt(sd_i * sd_n)
    defined = ok_i & ok_n & (denom > 0.0)
    n_def = int(defined.sum())
    if n_def < _int_at_least(min_defined, 1, "min_defined"):
        raise ValueError(f"fewer than {min_defined} times have non-degenerate ensemble dispersion")
    z = np.full(imp.shape[1], np.nan, dtype=np.float64)
    z[defined] = (mu_i[defined] - mu_n[defined]) / denom[defined]
    return {
        "label": "SYNTHETIC",
        "z": z,
        "mu_impacted": mu_i,
        "mu_control": mu_n,
        "sd_impacted": sd_i,
        "sd_control": sd_n,
        "n_defined": n_def,
        "n_times": int(imp.shape[1]),
    }


def classify_impact_regime(
    z: Array,
    *,
    decay_frac: float = 0.5,
    growth_slope: float = 0.0,
) -> str:
    """Label the Sec. 4.1 impact regime from the post-injection z path.

    ``dissipative`` when |z| decays below ``decay_frac`` of its peak by the
    window end (impact absorbed); ``non_dissipative`` when terminal |z| sits
    at/above the peak and the late-window OLS slope of z is positive beyond
    ``growth_slope`` (self-sustaining cascade); ``balanced`` otherwise (the
    paper's permanent, non-decaying critical boundary). Frozen-phase
    inputs with a near-zero pre-impact sigma should use ``melting`` from the
    caller's phase label instead — this classifier covers the continuous
    phase only (paper Fig. 3). Degenerate all-nan or empty z raises.
    """
    zz = np.asarray(z, dtype=np.float64).reshape(-1)
    zz = zz[np.isfinite(zz)]
    if zz.size < 3:
        raise ValueError("need at least 3 defined z values to classify")
    df = _prob(decay_frac, "decay_frac")
    gs = float(growth_slope)
    if not math.isfinite(gs):
        raise ValueError("growth_slope must be finite")
    peak = float(np.abs(zz).max())
    end = float(abs(zz[-1]))
    if peak <= 0.0:
        return "dissipative"  # no measurable displacement: fully absorbed
    if end < df * peak:
        return "dissipative"
    # Late-window (last third, at least 3 points) slope of z: a still-rising
    # impact is the self-sustaining non-dissipative cascade of paper Fig. 3.
    tail = zz[-max(3, zz.size // 3) :]
    if tail.size >= 3:
        t_idx = np.arange(tail.size, dtype=np.float64)
        slope = float(np.polyfit(t_idx, tail, 1)[0])
    else:
        slope = 0.0
    if end >= 0.8 * peak and slope > gs:
        return "non_dissipative"
    return "balanced"


def impact_zscore_experiment(
    *,
    config: ZILobConfig,
    size: int,
    scan: PhaseScanConfig | None = None,
    side: str = "buy",
    n_samples: int = 60,
) -> dict[str, Any]:
    """Run the paper's Sec. 4 impact protocol on the ZI-LOB.

    Per path: warm up ``scan.warmup`` seconds, record ``n_samples`` mids on
    the ``sample_interval`` grid, injecting ``size`` unit market orders of
    ``side`` instantaneously at the start of the recorded window (the
    paper's single aggressive order at ``t_Q``). Control paths omit the
    injection at matched seeds-offset. Returns the ensemble ``z(t)`` path,
    the classified impact regime, and per-path displacement statistics —
    the emergent complement to ``metaorder_impact_slope``'s log-log fit.

    ``size`` must stay modest relative to book depth or the injection wipes
    the touch: paths whose post-injection mid never becomes defined are
    dropped from the ensemble before scoring (fail-closed if all do).
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    q = _int_at_least(size, 1, "size")
    ns = _int_at_least(n_samples, 4, "n_samples")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    def _trace(path_seed: int, inject: bool) -> Array:
        sim = ZILobSimulator(replace(config, seed=path_seed))
        sim.run(sc.warmup)
        if inject:
            sim.inject_market_order(side, qty=q)  # type: ignore[arg-type]
        out = np.full(ns, np.nan, dtype=np.float64)
        t_next = sim.t
        i = 0
        while i < ns:
            sim.run(t_next)
            out[i] = _mid_in_ticks(sim)
            i += 1
            t_next += sc.sample_interval
        return out

    k = sc.n_paths
    impacted = np.stack([_trace(config.seed + i, True) for i in range(k)])
    control = np.stack([_trace(config.seed + 1000 + i, False) for i in range(k)])
    # Drop paths whose post-injection mid stays undefined (fully emptied
    # touch — a collapse realization, counted not silently scored).
    ok_i = np.isfinite(impacted).sum(axis=1) >= sc.min_defined_mids
    ok_c = np.isfinite(control).sum(axis=1) >= sc.min_defined_mids
    if not ok_i.any() or not ok_c.any():
        raise ValueError("post-injection book never re-formed a two-sided quote on all paths")
    n_dropped = int((~ok_i).sum() + (~ok_c).sum())
    res = impact_zscore(impacted[ok_i], control[ok_c], min_defined=2)
    regime = classify_impact_regime(res["z"])
    # First-to-last *defined* mid displacement per surviving path (edges can
    # be undefined while the touch is empty).
    disp = np.full(int(ok_i.sum()), np.nan, dtype=np.float64)
    for i, row in enumerate(impacted[ok_i]):
        fin = np.flatnonzero(np.isfinite(row))
        if fin.size >= 2:
            disp[i] = row[fin[-1]] - row[fin[0]]
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "z": res["z"].tolist(),
        "n_defined": res["n_defined"],
        "n_times": res["n_times"],
        "impact_regime": regime,
        "z_peak": float(np.nanmax(np.abs(res["z"]))),
        "z_final": float(res["z"][-1]) if np.isfinite(res["z"][-1]) else float("nan"),
        "size": q,
        "side": side,
        "n_paths_used_impacted": int(ok_i.sum()),
        "n_paths_used_control": int(ok_c.sum()),
        "n_paths_dropped": n_dropped,
        "mean_displacement_ticks": float(np.nanmean(disp))
        if np.isfinite(disp).any()
        else float("nan"),
        "claim": "simulator_internal_diagnostic_only",
    }


# ---------------------------------------------------------------------------
# Hysteresis + injection-phase clustering (lane supplements)
# ---------------------------------------------------------------------------


def hysteresis_sweep(
    *,
    base_config: ZILobConfig,
    lam_values: Sequence[float],
    scan: PhaseScanConfig | None = None,
    thick_levels: int = 8,
    thick_depth: int = 6,
) -> dict[str, Any]:
    """Initial-condition hysteresis across the liquidity axis.

    The paper's collapsed/continuous boundary is first-order: approaching it
    from opposite macroscopic states should show different transition points.
    The ZI-LOB has no agent memory, so the sweep direction is carried by the
    initial book: each ``lam`` is scanned twice — ``"cold"`` paths start
    from a thick seeded book (``thick_levels`` x ``thick_depth``, the
    liquid-phase initial condition) and ``"hot"`` paths from an empty book
    (the vapor-phase initial condition). ``loop_area`` integrates
    |sigma_cold - sigma_hot| over lam; ``lam_c_gap`` is the shift between
    the two directions' collapse boundaries. A nonzero loop is the
    first-order hysteresis signature; zero means a reversible transition.
    """
    if not isinstance(base_config, ZILobConfig):
        raise TypeError("base_config must be a ZILobConfig")
    sc = scan if scan is not None else PhaseScanConfig()
    if not isinstance(sc, PhaseScanConfig):
        raise TypeError("scan must be a PhaseScanConfig")
    lams = _seq_of_pos(lam_values, "lam_values")
    lv = _int_at_least(thick_levels, 1, "thick_levels")
    dp = _int_at_least(thick_depth, 1, "thick_depth")

    def _sweep(init_levels: int, init_depth: int) -> list[dict[str, Any]]:
        out = []
        for la in lams:
            cfg = replace(base_config, lam=la, init_levels=init_levels, init_depth=init_depth)
            out.append(ensemble_phase(cfg, sc))
        return out

    cold = _sweep(lv, dp)
    hot = _sweep(0, 0)
    sig_cold = np.asarray([c["sigma_step_median_ticks"] for c in cold], dtype=np.float64)
    sig_hot = np.asarray([c["sigma_step_median_ticks"] for c in hot], dtype=np.float64)
    col_cold = np.asarray([c["collapse_rate"] for c in cold], dtype=np.float64)
    col_hot = np.asarray([c["collapse_rate"] for c in hot], dtype=np.float64)
    both = np.isfinite(sig_cold) & np.isfinite(sig_hot)
    la = np.asarray(lams, dtype=np.float64)
    # Integrate the loop over the subset of grid points where BOTH sweeps
    # have a defined sigma (starved corner has no defined mid by design).
    loop_area = (
        float(np.trapezoid(np.abs(sig_cold[both] - sig_hot[both]), la[both], axis=0))
        if int(both.sum()) >= 2
        else float("nan")
    )
    lam_c_cold = _collapse_boundary(lams, col_cold, sc.collapse_rate_threshold)
    lam_c_hot = _collapse_boundary(lams, col_hot, sc.collapse_rate_threshold)
    lam_c_gap = (
        abs(lam_c_cold - lam_c_hot)
        if math.isfinite(lam_c_cold) and math.isfinite(lam_c_hot)
        else float("nan")
    )
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "lam_values": lams,
        "sigma_step_cold": sig_cold.tolist(),
        "sigma_step_hot": sig_hot.tolist(),
        "collapse_cold": col_cold.tolist(),
        "collapse_hot": col_hot.tolist(),
        "loop_area": loop_area,
        "lam_c_cold": lam_c_cold,
        "lam_c_hot": lam_c_hot,
        "lam_c_gap": lam_c_gap,
        "thick_levels": lv,
        "thick_depth": dp,
        "claim": "simulator_internal_diagnostic_only",
    }


def cluster_injection_phases(
    observables: Sequence[Mapping[str, Any]],
    *,
    collapse_empty_frac: float = 0.5,
    max_iter: int = 200,
) -> dict[str, Any]:
    """Deterministic phase clustering of per-path observables.

    Supplement to the scan cells: each entry supplies at least
    ``sigma_step_ticks`` and ``empty_time_frac`` (or ``collapse_rate``) from
    ``run_phase_path`` / ``ensemble_phase`` output. Labels follow
    ``classify_phase`` semantics: a path whose sigma is undefined or whose
    empty fraction reaches ``collapse_empty_frac`` is ``collapsed`` by
    construction (same logic as the scan, not re-clustered). The remaining
    points are split into two clusters by a deterministic 1-D Lloyd
    iteration on standardized sigma — centroids seeded at the min-sigma and
    max-sigma points, no randomness, bit-identical repeats. The low-sigma
    cluster is ``frozen``; the other is ``continuous_liquidity``. Fewer
    than 3 points, non-finite ``empty``, or a degenerate sigma raises.
    """
    if len(observables) < 3:
        raise ValueError(f"need at least 3 paths to cluster, got {len(observables)}")
    cth = _prob(collapse_empty_frac, "collapse_empty_frac")
    feats = np.empty((len(observables), 2), dtype=np.float64)
    for i, ob in enumerate(observables):
        if "sigma_step_ticks" not in ob:
            raise ValueError("each observable needs 'sigma_step_ticks'")
        s = float(ob["sigma_step_ticks"])
        e = float(ob.get("empty_time_frac", ob.get("collapse_rate", 0.0)))
        feats[i] = (s, e)
    if np.isinf(feats).any() or np.isnan(feats[:, 1]).any():
        raise ValueError("observables contain non-finite empty values")
    collapsed = np.isnan(feats[:, 0]) | (feats[:, 1] >= cth)
    labels = np.zeros(len(observables), dtype=np.intp)  # cluster 0 = collapsed
    n_iter = 0
    rest_idx = np.flatnonzero(~collapsed)
    centroids = np.empty((0, 2), dtype=np.float64)
    if rest_idx.size >= 2:
        # Frozen vs continuous splits on sigma alone — the same observable
        # classify_phase uses (empty only gates the collapsed pre-split).
        sub = feats[rest_idx, 0]
        mu = float(sub.mean())
        sd = float(sub.std())
        if sd <= 0.0:
            raise ValueError("degenerate feature (constant sigma)")
        z = (sub - mu) / sd
        c = np.array([z.min(), z.max()], dtype=np.float64)
        sub_lab = np.zeros(rest_idx.size, dtype=np.intp)
        for it in range(max_iter):
            n_iter = it + 1
            dist = (z[:, None] - c[None, :]) ** 2
            new_lab = dist.argmin(axis=1)
            new_c = c.copy()
            for k in range(2):
                members = z[new_lab == k]
                if members.size:
                    new_c[k] = members.mean()
            if np.array_equal(new_lab, sub_lab) and np.allclose(new_c, c):
                break
            sub_lab, c = new_lab, new_c
        # Low-sigma centroid -> frozen (cluster 1), high -> continuous (2).
        low = int(np.argmin(c))
        for i, lab in enumerate(sub_lab):
            labels[rest_idx[i]] = 1 if int(lab) == low else 2
        centroids = (c * sd + mu)[:, None]
    elif rest_idx.size == 1:
        labels[rest_idx[0]] = 2
    name_of = {0: "collapsed", 1: "frozen", 2: "continuous_liquidity"}
    return {
        "label": "SYNTHETIC",
        "data_source": AGENTIC_LOB_REVISION,
        "phase_labels": [name_of[int(k)] for k in labels],
        "cluster_ids": labels.tolist(),
        "centroids_raw": centroids.tolist(),
        "counts": {name_of[k]: int((labels == k).sum()) for k in range(3)},
        "n_iter": n_iter,
        "n_points": int(len(observables)),
        "n_collapsed_direct": int(collapsed.sum()),
    }


# ---------------------------------------------------------------------------
# Bench: one seeded, seconds-scale end-to-end diagnostic bundle
# ---------------------------------------------------------------------------


def bench_agentic_lob(*, seed: int = 7) -> dict[str, float]:
    """Seeded end-to-end bench returning a flat ``dict[str, float]``.

    Runs a small 3x3 phase diagram, one Sec.-4 z-score experiment, a planted
    two-state flow-regime boundary detection, and a sequential phase alarm
    on an exploding-volatility series. ~seconds; every value is a SYNTHETIC
    correctness diagnostic.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    out: dict[str, float] = {}
    sc = PhaseScanConfig(horizon=200.0, warmup=120.0, n_paths=3, seed=seed)
    base = santa_fe_config(seed=seed)

    diagram = phase_diagram(
        base_config=base,
        lam_values=(0.01, 0.06, 0.2),
        theta_values=(0.005, 0.02, 0.08),
        scan=sc,
    )
    cell = diagram["per_band"][0]["cells"]
    collapse_grid = np.asarray(diagram["per_band"][0]["collapse_grid"], dtype=np.float64)
    sigma_grid = np.asarray(diagram["per_band"][0]["sigma_step_grid"], dtype=np.float64)
    out["diagram_collapse_rate_max"] = float(np.max(collapse_grid))
    out["diagram_collapse_rate_min"] = float(np.min(collapse_grid))
    out["diagram_sigma_max_ticks"] = float(np.nanmax(sigma_grid))
    out["diagram_n_cells"] = float(diagram["n_cells_total"])
    lam_c = np.asarray(diagram["per_band"][0]["lam_c_by_theta"], dtype=np.float64)
    out["diagram_lam_c_median"] = (
        float(np.nanmedian(lam_c)) if np.isfinite(lam_c).any() else float("nan")
    )
    phase_flat = [str(p["phase"]) for row in cell for p in row]
    out["diagram_n_collapsed"] = float(sum(p == "collapsed" for p in phase_flat))
    out["diagram_n_frozen"] = float(sum(p == "frozen" for p in phase_flat))
    out["diagram_n_continuous"] = float(sum(p == "continuous_liquidity" for p in phase_flat))

    zres = impact_zscore_experiment(config=base, size=12, scan=sc, n_samples=40)
    out["impact_z_peak"] = float(zres["z_peak"])
    out["impact_n_defined"] = float(zres["n_defined"])
    out["impact_regime_code"] = float(IMPACT_REGIMES.index(zres["impact_regime"]))

    flow = MarkovRegimeFlow(
        (
            RegimeState("calm", 1.0, 0.5),
            RegimeState("storm", 4.0, 0.5),
        ),
        (0.98, 0.9),
        seed=seed + 5,
    )
    feats = phase_feature_series(base, sc, bucket_seconds=10.0, flow=flow)
    out["feature_n_buckets"] = float(feats["n_buckets"])
    out["feature_empty_frac_max"] = float(np.max(feats["empty_frac"]))
    # Planted two-level feature step: deterministic boundary at the jump.
    rng_plant = np.random.default_rng(seed + 11)
    planted = {
        "mid_sigma_ticks": np.concatenate(
            [
                0.02 + 0.005 * rng_plant.standard_normal(30),
                0.9 + 0.05 * rng_plant.standard_normal(30),
            ]
        ),
        "empty_frac": np.zeros(60),
    }
    det = detect_phase_boundaries(planted, method="binseg", min_segment=3)
    out["boundaries_n"] = float(len(det["boundaries"]))
    out["segments_n"] = float(det["n_segments"])

    rng = np.random.default_rng(seed)
    calm = rng.normal(0.2, 0.05, size=40)
    storm = rng.normal(2.0, 0.4, size=40)
    alarm = phase_alarm(np.concatenate([calm, storm]), calibration_size=30, seed=seed)
    out["alarm_wealth_final"] = float(alarm["wealth_final"])
    out["alarm_fired"] = 1.0 if alarm["alarm"] else 0.0
    out["alarm_sr_final"] = float(alarm["shiryaev_roberts_final"])
    return out
