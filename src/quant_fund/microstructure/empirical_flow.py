"""empirical_flow — tape-calibrated ZI-LOB configuration.

Builds a :class:`ZILobConfig` whose arrival rates and event-size tables
are resampled from the committed real-tape receipts via
:mod:`tape_stats` — the nonparametric complement to ``split_flow``
(which keeps parametric Pareto splitting). The gaps this closes are the
expressivity divergences measured by the wave-21b lanes: unit-size
events (``round_lot`` gap +0.37), no multi-level sweeps
(``sweep_width`` structurally zero), and an arrival mix that modeled
only the 3% of tape events that execute.

Calibration is explicit and honest: rates map linearly
(``lam = sub/(2·band)``, ``mu = exec/2``), ``theta_cxl`` is solved by a
seeded probe run so the *steady-state* cancel:submission ratio matches
the tape, and sizes are scaled into sim units by ``size_scale`` (sim
unit ≠ share; the scale is reported, not hidden).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from quant_fund.microstructure.tape_stats import TapeStats, scaled_size_pmf
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig, ZILobSimulator

EMPIRICAL_FLOW_SCHEMA = "empirical_flow.v1"


@dataclass(frozen=True)
class EmpiricalCalibration:
    """The solved calibration knobs (reported, not assumed)."""

    lam: float
    mu: float
    theta_cxl: float
    band: int
    size_scale: float
    probe_depth: float
    probe_horizon: int


def _pos(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def calibrate_theta(
    stats: TapeStats,
    *,
    lam: float,
    mu: float,
    band: int,
    seed: int,
    probe_horizon: int = 4000,
) -> tuple[float, float]:
    """Solve ``theta_cxl`` so steady-state delete:sub matches the tape.

    A seeded probe run measures the equilibrium resting depth ``D``;
    the tape's delete rate then maps to ``theta = delete / D``. Returns
    ``(theta_cxl, probe_depth)``. Fail-closed on a degenerate probe.
    """
    _pos(lam, "lam")
    _pos(mu, "mu")
    if probe_horizon < 100:
        raise ValueError(f"probe_horizon must be >= 100, got {probe_horizon!r}")
    probe = ZILobSimulator(
        ZILobConfig(lam=lam, mu=mu, theta_cxl=0.01, band=band, seed=seed, init_depth=2)
    )
    depths: list[float] = []
    for _ in range(probe_horizon):
        probe.step()
        if probe.n_events % 8 == 0:
            depths.append(float(probe.total_depth))
    if not depths:
        raise ValueError("probe run produced no depth samples")
    tail = depths[len(depths) // 2 :]
    d = sum(tail) / len(tail)
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError(f"probe run yielded degenerate depth {d!r}")
    delete_rate = stats.rate("delete") + stats.rate("cxl_part")
    theta = delete_rate / d
    return float(theta), float(d)


def empirical_zi_config(
    stats: TapeStats,
    *,
    seed: int = 0,
    band: int = 20,
    size_scale: float = 1.0 / 30.0,
    probe_horizon: int = 4000,
) -> tuple[ZILobConfig, EmpiricalCalibration]:
    """ZI-LOB config calibrated to ``stats`` (nonparametric sizes).

    ``band`` widens the placement support toward the tape's observed
    spread occupancy (a band below ~13 ticks cannot express the 9-21
    tick book the AMZN tape occupies). ``size_scale`` converts shares
    to sim units — the sim's unit order is a modeling unit, and the
    scale is reported in the receipt.
    """
    if not isinstance(stats, TapeStats):
        raise ValueError(f"stats must be a TapeStats, got {type(stats)!r}")
    _pos(size_scale, "size_scale")
    if isinstance(band, bool) or int(band) < 1:
        raise ValueError(f"band must be an int >= 1, got {band!r}")
    band = int(band)
    sub_rate = stats.rate("sub")
    exec_rate = stats.rate("exec") + stats.rate("exec_hidden")
    lam = sub_rate / (2.0 * band)
    mu = exec_rate / 2.0
    theta, probe_depth = calibrate_theta(
        stats, lam=lam, mu=mu, band=band, seed=seed, probe_horizon=probe_horizon
    )
    pmf = scaled_size_pmf(stats.size_pmf, size_scale)
    cfg = ZILobConfig(
        lam=lam,
        mu=mu,
        theta_cxl=theta,
        band=band,
        p_buy=0.5,
        init_levels=3,
        init_depth=3,
        mo_size_pmf=pmf,
        lo_size_pmf=pmf,
        seed=seed,
    )
    calib = EmpiricalCalibration(
        lam=lam,
        mu=mu,
        theta_cxl=theta,
        band=band,
        size_scale=float(size_scale),
        probe_depth=probe_depth,
        probe_horizon=probe_horizon,
    )
    return cfg, calib


def load_empirical_config(
    receipts_root: str | Path,
    *,
    ticker: str = "amzn",
    seed: int = 0,
    **kwargs: object,
) -> tuple[ZILobConfig, EmpiricalCalibration, TapeStats]:
    """Load receipts → stats → calibrated config in one call."""
    from quant_fund.microstructure.tape_stats import load_tape_stats

    stats = load_tape_stats(receipts_root, ticker)
    cfg, calib = empirical_zi_config(stats, seed=seed, **kwargs)  # type: ignore[arg-type]
    return cfg, calib, stats
