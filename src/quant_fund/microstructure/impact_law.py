"""Metaorder impact law: does the ZI-LOB reproduce square-root impact?

The Donier et al. (2015) latent-liquidity result is that a metaorder of
size ``Q`` moves price as ``I(Q) ~ Q^psi`` with ``psi ~ 0.5`` — *not*
the naive linear Kyle picture. The simulator's ``density_exponent``
knob claims both regimes:

- ``beta=0``: flat book density -> cumulative liquidity proportional to
  depth walked -> linear impact (``psi ~ 1``).
- ``beta=1``: triangular density rising with distance -> cumulative
  liquidity ~ d^2 -> square-root impact (``psi ~ 0.5``).

This module *measures* the exponent rather than trusting the docstring:
sweep ``Q``, record the mean price move a market order of that size
produces (arrival mid -> volume-weighted fill price), then fit
``log I = log c + psi log Q`` on the sweep. A seeded warm-up leaves the
book in its stationary state before each probe.

Two measurements per (Q, beta): the instantaneous impact (avg fill vs
arrival mid) and — in the spirit of the volume-diffusion theory — the
*peak* level walked. Evidence class: SYNTHETIC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

IMPACT_LAW_SCHEMA = "impact_law.v1"


def _flow(seed: int) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="busy", intensity_mult=1.3, p_buy=0.5),
        ],
        stay_probs=[0.9, 0.9],
        seed=seed,
    )


def _warm_sim(density_exponent: float, seed: int, warmup: int) -> ZILobSimulator:
    sim = ZILobSimulator(
        ZILobConfig(seed=seed, density_exponent=density_exponent),
        flow=_flow(seed),
    )
    for _ in range(warmup):
        sim.step()
    return sim


def probe_impact(
    sim: ZILobSimulator, *, side: Literal["buy", "sell"], qty: int
) -> tuple[float, int]:
    """Fire a qty market order; return (impact_in_ticks, levels_walked).

    Impact is the VWAP slippage vs the arrival mid — the standard
    metaorder-impact measure. Returns NaN ticks when the order cannot
    fill at all.
    """
    if qty < 1:
        raise ValueError("qty must be >= 1")
    arrival = sim.mid
    if arrival is None:
        raise ValueError("book empty at arrival")
    tick = sim.cfg.tick
    trades = sim.inject_market_order(side, qty)
    if not trades:
        return float("nan"), 0
    # terminal impact: last fill vs arrival mid — the metaorder-impact
    # measure; VWAP dilutes the walk by construction
    terminal = trades[-1].price
    sign = 1.0 if side == "buy" else -1.0
    max_walk = max(abs(t.price - arrival) for t in trades)
    return sign * (terminal - arrival) / tick, int(math.ceil(max_walk / tick))


@dataclass(frozen=True)
class ImpactPoint:
    qty: int
    impact_ticks_mean: float
    impact_ticks_std: float
    levels_walked_mean: float
    n_trials: int


def impact_curve(
    *,
    qty_grid: tuple[int, ...],
    density_exponent: float,
    n_trials: int,
    warmup: int = 300,
    pace_between: int = 60,
) -> list[ImpactPoint]:
    """Mean instantaneous impact at each parent size on one density regime."""
    if not qty_grid or min(qty_grid) < 1:
        raise ValueError("qty_grid must be non-empty positive ints")
    pts: list[ImpactPoint] = []
    for qi, qty in enumerate(qty_grid):
        impacts: list[float] = []
        walks: list[int] = []
        for k in range(n_trials):
            sim = _warm_sim(density_exponent, seed=1000 * qi + k, warmup=warmup)
            imp, walk = probe_impact(sim, side="buy", qty=qty)
            if math.isfinite(imp):
                impacts.append(imp)
                walks.append(walk)
        if not impacts:
            raise ValueError(f"no fills at qty={qty} — book cannot absorb the probe")
        pts.append(
            ImpactPoint(
                qty=qty,
                impact_ticks_mean=float(np.mean(impacts)),
                impact_ticks_std=float(np.std(impacts)),
                levels_walked_mean=float(np.mean(walks)),
                n_trials=len(impacts),
            )
        )
    return pts


def fit_power_law(points: list[ImpactPoint]) -> tuple[float, float, float]:
    """Fit I(Q) = c*Q^psi on (log Q, log I); return (psi, c, r2)."""
    if len(points) < 3:
        raise ValueError("need >= 3 impact points for a power-law fit")
    q = np.asarray([p.qty for p in points], dtype=float)
    i = np.asarray([p.impact_ticks_mean for p in points], dtype=float)
    if np.any(i <= 0):
        raise ValueError("non-positive impact — log fit impossible")
    lx, ly = np.log(q), np.log(i)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - float(np.sum(resid * resid)) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(math.exp(intercept)), r2


def impact_law_bench(
    *,
    qty_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    n_trials: int = 8,
    warmup: int = 300,
) -> dict[str, Any]:
    """Fit the impact exponent under flat vs triangular book density.

    The Donier prediction: psi ~ 1 under ``density_exponent=0``,
    psi ~ 0.5 under ``density_exponent=1``. The bench reports both
    fitted exponents with the regression r2, and asserts nothing about
    the absolute level — only the scaling exponent, which is what the
    volume-diffusion theory actually predicts.
    """
    cells: dict[str, Any] = {}
    for beta in (0.0, 1.0):
        pts = impact_curve(
            qty_grid=qty_grid, density_exponent=beta, n_trials=n_trials, warmup=warmup
        )
        psi, c, r2 = fit_power_law(pts)
        # sub-saturation fit: only points that didn't exhaust the 5-level
        # band (walk < band) — the clean scaling region
        subs = [p for p in pts if p.levels_walked_mean < 5.0]
        psi_s = c_s = r2_s = float("nan")
        if len(subs) >= 3:
            psi_s, c_s, r2_s = fit_power_law(subs)
        cells[f"beta_{beta:.0f}"] = {
            "psi_hat": psi,
            "prefactor": c,
            "fit_r2": r2,
            "psi_subsat": psi_s,
            "prefactor_subsat": c_s,
            "fit_r2_subsat": r2_s,
            "n_subsat_points": len(subs),
            "curve": [
                {
                    "qty": p.qty,
                    "impact_ticks_mean": p.impact_ticks_mean,
                    "impact_ticks_std": p.impact_ticks_std,
                    "levels_walked_mean": p.levels_walked_mean,
                    "n_trials": p.n_trials,
                }
                for p in pts
            ],
        }
    payload = {
        "schema": IMPACT_LAW_SCHEMA,
        "kind": "impact_law",
        "qty_grid": list(qty_grid),
        "n_trials": n_trials,
        "warmup_steps": warmup,
        "cells": cells,
        "prediction": (
            "psi~1 under beta=0 (flat book), psi~0.5 under beta=1 (Donier); "
            "the 5-level placement band saturates terminal impact at large Q, "
            "so psi_subsat (walk < band) is the cleaner scaling estimate"
        ),
        "psi_ordering_holds": cells["beta_0"]["psi_hat"] > cells["beta_1"]["psi_hat"],
        "subsat_ordering_holds": cells["beta_0"]["psi_subsat"] > cells["beta_1"]["psi_subsat"]
        if cells["beta_0"]["n_subsat_points"] >= 3 and cells["beta_1"]["n_subsat_points"] >= 3
        else None,
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
