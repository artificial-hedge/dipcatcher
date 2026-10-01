"""Response-surface map for the ZI-LOB simulator (lane ``sim_sensitivity``).

Maps the simulator's own exogenous flow parameters — ``lam`` (limit-order
rate per side per level), ``mu`` (market-order rate per side), and
``theta_cxl`` (per-resting-order cancellation rate) — onto the emergent
microstructure statistics they produce: market-order share of events, order-
flow sign autocorrelation, spread, mid diffusion, and top-of-book depth.

This is **calibration guidance for the simulator**, not market evidence: it
answers "which knob moves which stylized fact" inside the sim's parameter
space so downstream lanes (market-making sessions, regime-flow experiments)
know what to turn to hit a target regime. Every number is a labeled
SYNTHETIC correctness diagnostic of ``microstructure.zi_lob_simulator``
itself; there is no broker connectivity or live-trading claim anywhere.

Two flow arms per grid cell share one engine: ``iid`` (``flow=None``,
stationary Poisson MO arrivals) and ``markov`` (a two-state
``MarkovRegimeFlow`` modulating MO intensity and direction on the MO clock,
cf. Moret & Lillo 2026 Sec. 5). Comparing arms separates parameter effects
from flow-regime effects.

``sim_sensitivity_bench`` seals the grid into ``receipts/sim_sensitivity.json``
(kind ``sim_sensitivity``, schema ``sim_sensitivity.v1``,
``receipt_sha256`` over the canonical JSON — the fleet_eval seal convention).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
    order_flow_autocorrelation,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SIM_SENSITIVITY_KIND = "sim_sensitivity"
SIM_SENSITIVITY_SCHEMA = "sim_sensitivity.v1"
RECEIPT_PATH = Path("receipts/sim_sensitivity.json")

#: The response surface: a 3×3×3 full-factorial over the three exogenous
#: flow rates (per simulated second). Santa Fe baseline is the center cell.
GRID_LAM = (0.03, 0.06, 0.12)
GRID_MU = (0.05, 0.10, 0.20)
GRID_THETA_CXL = (0.01, 0.02, 0.04)
GRID_PARAMS = ("lam", "mu", "theta_cxl")

#: Markov arm: calm/bursty two-state chain on the MO clock (bursty triples MO
#: intensity and skews buy probability; sticky states ~200/~67 MOs).
MARKOV_ARM_STATES = (
    RegimeState("calm", 1.0, 0.5),
    RegimeState("bursty", 3.0, 0.62),
)
MARKOV_ARM_STAY = (0.995, 0.985)

ARMS = ("iid", "markov")
STAT_NAMES = (
    "mo_fraction",
    "sign_lag1",
    "spread_ticks_median",
    "mid_move_std",
    "mean_top_depth",
)

INTERPRETATION = (
    "Sensitivity map of the ZI-LOB simulator's own parameter space. "
    "drivers.<arm>.<stat> names the flow parameter with the largest |Spearman "
    "rho| for that statistic across the 27-cell grid — calibration guidance "
    "for which knob moves which stylized fact. iid = stationary Poisson flow; "
    "markov = two-state Markov-modulated MO clock (calm/bursty). One seeded "
    "path per cell-arm; point estimates, not confidence intervals. SYNTHETIC "
    "correctness diagnostic of the simulator, never market evidence."
)


def _pos_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return int(value)


def sim_stats(cfg: ZILobConfig, flow: MarkovRegimeFlow | None, horizon: int) -> dict[str, Any]:
    """Microstructure statistics of one seeded ZI-LOB path.

    ``horizon`` counts *event steps* (``sim.step()`` calls), not simulated
    seconds. All quote-derived series sample the book once per step, only on
    steps where both best quotes exist. Statistics that are undefined for the
    realized path (e.g. sign autocorrelation of a constant or too-short trade
    stream) come back ``None`` — an honest "unmeasured", never a fabricated
    number.

    - ``mo_fraction``: market-order executions per event step —
      ``len(sim.trades) / horizon`` (trades, not MO arrivals; arrivals that
      hit an empty side are no-ops and do not trade).
    - ``sign_lag1``: lag-1 autocorrelation of the aggressor sign stream
      (+1 buy / -1 sell) via ``order_flow_autocorrelation``.
    - ``spread_ticks_median``: median quoted spread in ticks over two-sided
      steps.
    - ``mid_move_std``: population std of successive mid moves, in ticks.
    - ``mean_top_depth``: mean displayed size at the touch (best-bid depth +
      best-ask depth) over two-sided steps.
    - ``n_trades``: executions observed (context for the stats above).
    """
    if not isinstance(cfg, ZILobConfig):
        raise TypeError("cfg must be a ZILobConfig")
    h = _pos_int(horizon, "horizon")
    sim = ZILobSimulator(cfg, flow=flow)
    spreads: list[float] = []
    mids: list[float] = []  # mid level in ticks; level 0 is the seed mid s0
    top_depths: list[float] = []
    for _ in range(h):
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        if bb is None or ba is None:
            continue
        spreads.append(float(ba - bb))
        mids.append(0.5 * float(bb + ba))
        top_depths.append(float(sim.depth_at("buy", bb) + sim.depth_at("sell", ba)))

    signs = [1.0 if trade.aggressor == "buy" else -1.0 for trade in sim.trades]
    sign_lag1: float | None = None
    if len(signs) >= 3:
        try:
            sign_lag1 = float(order_flow_autocorrelation(signs, lag=1))
        except ValueError:
            sign_lag1 = None  # constant sign stream: autocorrelation undefined

    return {
        "label": "SYNTHETIC",
        "mo_fraction": len(sim.trades) / h,
        "sign_lag1": sign_lag1,
        "spread_ticks_median": float(np.median(spreads)) if spreads else None,
        "mid_move_std": float(np.std(np.diff(mids))) if len(mids) >= 2 else None,
        "mean_top_depth": float(np.mean(top_depths)) if top_depths else None,
        "n_trades": len(sim.trades),
    }


def _markov_arm(seed: int) -> MarkovRegimeFlow:
    """The grid's Markov-modulated MO-clock arm (fresh chain per cell)."""
    return MarkovRegimeFlow(MARKOV_ARM_STATES, MARKOV_ARM_STAY, seed=seed)


def _dominant_drivers(cells: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Per-statistic dominant grid parameter by |Spearman rho| over the cells.

    Cells where the statistic is ``None`` (unmeasured) are dropped pairwise;
    a constant statistic yields an undefined rho (``None``) and cannot win.
    """
    params = {
        name: np.asarray([float(cell[name]) for cell in cells], dtype=np.float64)
        for name in GRID_PARAMS
    }
    out: dict[str, Any] = {}
    for stat in STAT_NAMES:
        values = np.asarray(
            [float(cell[arm][stat]) if cell[arm][stat] is not None else math.nan for cell in cells],
            dtype=np.float64,
        )
        mask = np.isfinite(values)
        rhos: dict[str, float | None] = {}
        for name, series in params.items():
            if int(mask.sum()) < 3:
                rhos[name] = None
                continue
            rho = float(spearmanr(series[mask], values[mask]).statistic)
            rhos[name] = rho if math.isfinite(rho) else None
        defined = {name: rho for name, rho in rhos.items() if rho is not None}
        top = max(defined, key=lambda name: abs(defined[name])) if defined else None
        out[stat] = {
            "param": top,
            "rho": defined[top] if top is not None else None,
            "rhos": rhos,
            "n_points": int(mask.sum()),
        }
    return out


def sensitivity_grid(*, horizon: int = 8000, seed: int = 7) -> dict[str, Any]:
    """Response surface of ``sim_stats`` over the (lam, mu, theta_cxl) grid.

    Every cell runs both flow arms at the same ``seed``: ``iid`` stationary
    flow and ``markov`` two-state-modulated flow. ``drivers`` reports, per
    arm and statistic, the parameter with the largest |Spearman rho| across
    the grid — the dominant driver of that stylized fact.
    """
    h = _pos_int(horizon, "horizon")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    cells: list[dict[str, Any]] = []
    for lam in GRID_LAM:
        for mu in GRID_MU:
            for theta_cxl in GRID_THETA_CXL:
                cfg = ZILobConfig(lam=lam, mu=mu, theta_cxl=theta_cxl, seed=seed)
                cells.append(
                    {
                        "lam": lam,
                        "mu": mu,
                        "theta_cxl": theta_cxl,
                        "iid": sim_stats(cfg, None, h),
                        "markov": sim_stats(cfg, _markov_arm(seed), h),
                    }
                )
    return {
        "label": "SYNTHETIC",
        "horizon": h,
        "seed": seed,
        "grid": {"lam": list(GRID_LAM), "mu": list(GRID_MU), "theta_cxl": list(GRID_THETA_CXL)},
        "arms": list(ARMS),
        "markov_arm": {
            "states": [
                {"name": s.name, "intensity_mult": s.intensity_mult, "p_buy": s.p_buy}
                for s in MARKOV_ARM_STATES
            ],
            "stay_probs": list(MARKOV_ARM_STAY),
        },
        "cells": cells,
        "drivers": {arm: _dominant_drivers(cells, arm) for arm in ARMS},
    }


def _sanitize(value: Any) -> Any:
    """Replace non-finite floats with ``None`` recursively (strict-JSON safe)."""
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def sim_sensitivity_bench(*, horizon: int = 8000, seed: int = 7) -> dict[str, Any]:
    """Sealed ``sim_sensitivity.v1`` receipt over the response surface.

    The seal is ``receipt_sha256 = sha256(canonical_json(payload))`` — the
    fleet_eval convention: the digest binds the canonical bytes of everything
    except itself, and strict-JSON verifiers can re-derive it.
    """
    payload: dict[str, Any] = {
        "kind": SIM_SENSITIVITY_KIND,
        "schema": SIM_SENSITIVITY_SCHEMA,
        "data_label": "SYNTHETIC",
        "research_only": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        **sensitivity_grid(horizon=horizon, seed=seed),
        "interpretation": INTERPRETATION,
    }
    payload = _sanitize(payload)
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


def write_sim_sensitivity_receipt(
    path: Path | str = RECEIPT_PATH,
    *,
    horizon: int = 8000,
    seed: int = 7,
) -> Path:
    """Run the bench and write the sealed receipt as strict sorted JSON."""
    payload = sim_sensitivity_bench(horizon=horizon, seed=seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return out


__all__ = [
    "ARMS",
    "GRID_LAM",
    "GRID_MU",
    "GRID_PARAMS",
    "GRID_THETA_CXL",
    "INTERPRETATION",
    "RECEIPT_PATH",
    "SIM_SENSITIVITY_KIND",
    "SIM_SENSITIVITY_SCHEMA",
    "STAT_NAMES",
    "sensitivity_grid",
    "sim_sensitivity_bench",
    "sim_stats",
    "write_sim_sensitivity_receipt",
]
