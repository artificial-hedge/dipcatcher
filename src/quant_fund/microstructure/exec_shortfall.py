"""Implementation-shortfall measurement on the ZI-LOB.

Executes a parent order on the simulator book following a given
schedule and compares realized slippage against the arrival mid.

Two schedules are compared:

- ``twap``: uniform child sizes — the risk-neutral AC optimum under
  permanent impact only.
- ``ac_front``: the Almgren-Chriss exponential front-load
  ``x_k = X * sinh(kappa * (T - t_k)) / sinh(kappa * T)`` — trading
  faster early to cut timing risk under non-zero risk aversion.

The exercise is deliberately honest about the ZI sim: its book has no
true permanent impact (the replenish process is memoryless), so the
empirical question is how much slippage the transient book walk costs
and whether front-loading actually reduces IS variance across seeds —
not whether the sim reproduces stylized square-root impact. Results
are labeled SYNTHETIC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

EXEC_SHORTFALL_SCHEMA = "exec_shortfall.v1"

Schedule = Literal["twap", "ac_front"]


def ac_trajectory(parent_qty: int, n_slices: int, *, kappa: float) -> NDArray[np.int64]:
    """Integer child sizes following the AC sinh front-load, summing to X."""
    if parent_qty < 1 or n_slices < 1:
        raise ValueError("parent_qty and n_slices must be >= 1")
    if parent_qty < n_slices:
        raise ValueError("parent_qty must be >= n_slices (unit lots)")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be a positive finite number")
    t = np.arange(n_slices + 1, dtype=float)
    hold = parent_qty * np.sinh(kappa * (n_slices - t)) / math.sinh(kappa * n_slices)
    hold[-1] = 0.0
    # cumulative rounding keeps every child an integer and sums to X
    cume = np.round(parent_qty - hold).astype(np.int64)
    cume[0], cume[-1] = 0, parent_qty
    return np.diff(cume)


def twap_trajectory(parent_qty: int, n_slices: int) -> NDArray[np.int64]:
    """Uniform child sizes; remainder goes to the last slice."""
    if parent_qty < 1 or n_slices < 1:
        raise ValueError("parent_qty and n_slices must be >= 1")
    if parent_qty < n_slices:
        raise ValueError("parent_qty must be >= n_slices (unit lots)")
    base = parent_qty // n_slices
    out = np.full(n_slices, base, dtype=np.int64)
    out[-1] += parent_qty - base * n_slices
    return out


@dataclass(frozen=True)
class ExecResult:
    avg_price: float
    arrival_mid: float
    is_ticks: float  # signed implementation shortfall in tick units
    n_fills: int
    n_lots_filled: int
    max_level_walked: int


def run_execution(
    sim: ZILobSimulator,
    *,
    side: Literal["buy", "sell"],
    schedule: NDArray[np.int64],
    pace: int,
) -> ExecResult:
    """Execute ``schedule`` on the sim, pacing ``pace`` event-steps between slices."""
    if pace < 1:
        raise ValueError("pace must be >= 1")
    arrival = sim.mid
    if arrival is None:
        raise ValueError("book is empty at arrival — cannot measure shortfall")
    tick = sim.cfg.tick
    fill_prices: list[float] = []
    max_walk = 0
    for qty in schedule:
        for _ in range(pace):
            sim.step()
        trades = sim.inject_market_order(side, int(qty))
        for tr in trades:
            fill_prices.append(tr.price)
            walk = abs(tr.price - arrival) / tick
            max_walk = max(max_walk, int(walk))
    if not fill_prices:
        raise ValueError("execution produced no fills — book never replenished")
    avg = float(np.mean(fill_prices))
    sign = 1.0 if side == "buy" else -1.0
    return ExecResult(
        avg_price=avg,
        arrival_mid=arrival,
        is_ticks=sign * (avg - arrival) / tick,
        n_fills=len(fill_prices),
        n_lots_filled=int(sum(int(q) for q in schedule)),
        max_level_walked=max_walk,
    )


def _two_state_flow(seed: int, *, p_buy_trend: float) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend", intensity_mult=1.4, p_buy=p_buy_trend),
        ],
        stay_probs=[0.92, 0.85],
        seed=seed,
    )


def exec_shortfall_bench(
    *,
    parent_qty: int = 60,
    n_slices: int = 12,
    pace: int = 40,
    n_seeds: int = 12,
    kappa: float = 0.35,
) -> dict[str, Any]:
    """Compare TWAP vs AC-front-loaded execution shortfall across seeds.

    Both schedules trade the same parent on the same seeded flow; the
    paired-seed difference isolates schedule effect from tape luck.
    Runs under both book-density regimes: ``density_exponent=0`` (flat
    book, linear impact — AC's own assumption class) and ``1``
    (triangular book, square-root impact per Donier et al. 2015 —
    where the linear-AC trajectory is no longer optimal).
    """
    schedules: dict[str, NDArray[np.int64]] = {
        "twap": twap_trajectory(parent_qty, n_slices),
        "ac_front": ac_trajectory(parent_qty, n_slices, kappa=kappa),
    }
    density_cells: dict[str, Any] = {}
    for beta in (0.0, 1.0):
        per_sched: dict[str, Any] = {}
        is_mat: dict[str, list[float]] = {k: [] for k in schedules}
        for name, sched in schedules.items():
            rows = []
            for k in range(n_seeds):
                sim = ZILobSimulator(
                    ZILobConfig(seed=k, density_exponent=beta),
                    flow=_two_state_flow(k, p_buy_trend=0.65),
                )
                # warm the book before arrival
                for _ in range(400):
                    sim.step()
                res = run_execution(sim, side="buy", schedule=sched, pace=pace)
                rows.append(res.is_ticks)
                is_mat[name].append(res.is_ticks)
            arr = np.asarray(rows)
            per_sched[name] = {
                "is_ticks_mean": float(np.mean(arr)),
                "is_ticks_median": float(np.median(arr)),
                "is_ticks_q95": float(np.quantile(arr, 0.95)),
                "is_ticks_std": float(np.std(arr)),
                "worst_fill_ticks": float(np.max(arr)),
            }
        diff = np.asarray(is_mat["ac_front"]) - np.asarray(is_mat["twap"])
        density_cells[f"beta_{beta:.0f}"] = {
            "schedules": per_sched,
            "paired_ac_minus_twap": {
                "mean": float(np.mean(diff)),
                "median": float(np.median(diff)),
                "frac_better": float(np.mean(diff < 0)),
            },
        }
    payload = {
        "schema": EXEC_SHORTFALL_SCHEMA,
        "kind": "exec_shortfall",
        "parent_qty": parent_qty,
        "n_slices": n_slices,
        "pace_steps": pace,
        "n_seeds": n_seeds,
        "kappa": kappa,
        "side": "buy",
        "density_cells": density_cells,
        "interpretation": (
            "beta_0 (flat book): linear transient impact — AC front-loading "
            "shifts timing risk only. beta_1 (triangular book): square-root "
            "impact — the linear-AC schedule is provably suboptimal there; "
            "the paired contrast measures how much."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
