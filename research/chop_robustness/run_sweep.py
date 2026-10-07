"""Pre-registered chop-regime robustness sweep (docs/SIM_LIVE_PNL.md).

The pre-registration block in docs/SIM_LIVE_PNL.md is dated 2026-10-07T17:05
+05:30 and fixes the mechanisms, parameterizations, windows and decision
rules. This driver runs EXACTLY that declaration and reports every
mechanism x window cell, losers included.

Honesty: retrospective analytics only (simulated books through run_backtest on
collected Binance USDT spot bars, modeled costs), carried with
live_pnl_claim=false. Binance USDT spot majors only; not market evidence; not
a live-trading or profitability claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.config.models import AppConfig
from quant_fund.paper.quantile_signals import QuantilePolicy
from quant_fund.paper.sim_live import StrategySlot, run_sim_live

SPEC = "vincent(ewma_emp+ewma_emp94+ewma_emp99)"

BASE_POLICY: dict[str, Any] = {
    "mode": "long_flat",
    "kappa": 0.15,
    "gross_target": 2.0,
    "name_cap": 0.5,
    "cost_gate": 0.15,
    "deadband": 0.01,
    "band_lo": 0.05,
    "band_hi": 0.95,
    "gate_on": "edge",
    "sizing": "risk",
    "book_vol_target": 0.03,
    "tail_gate": None,
    "persist_bars": 1,
    "exit_persist": 3,
    "mkt_disp_cut": None,
    "top_k": None,
    "mkt_edge_min": None,
    "leader_sid": "BTCUSDT",
    "leader_edge_min": 0.0,
    "w_alpha": 1.0,
    "gate_out": None,
    "meta_min": None,
    "meta_lookback": 60,
    "accel_min": None,
    "rebal_every": 1,
    "breadth_gross": False,
    "fund_cut": None,
}

MECHANISMS: dict[str, dict[str, Any]] = {
    "vol_scale_002": {"book_vol_target": 0.02},
    "breadth_gross": {"breadth_gross": True},
    "trend_gate_010": {"mkt_edge_min": 0.10},
}

MAJORS_5 = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
MAJORS_3 = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]

WINDOWS: dict[str, dict[str, Any]] = {
    "W1_full": {"symbols": MAJORS_5, "eval_tail_bars": None},
    "W2_last800d": {"symbols": MAJORS_5, "eval_tail_bars": 800},
    "W3_last500d": {"symbols": MAJORS_5, "eval_tail_bars": 500},
    "W4_2018_05_8p3y": {"symbols": MAJORS_3, "eval_tail_bars": None},
}

OUT_DIR = Path("research/chop_robustness_20261007b")  # re-run under the bench config


def _slot(name: str, overrides: dict[str, Any]) -> StrategySlot:
    return StrategySlot(name=name, spec=SPEC, policy=QuantilePolicy(**{**BASE_POLICY, **overrides}))


def _slots() -> list[StrategySlot]:
    return [_slot("base", {}), *[_slot(name, diff) for name, diff in MECHANISMS.items()]]


def _bench_config() -> AppConfig:
    """The lane's bench config (binance_public_data source), not AppConfig() defaults.

    The first sweep run (research/chop_robustness_20261007/) used AppConfig()
    defaults, whose data_label resolves to 'synthetic' and whose risk/execution
    defaults left the base book degenerate (0 fills) on 3 of 4 windows. This
    re-run pins the source to the lane's bench receipts' 'binance_public_data'
    so the base book can be checked against the historical receipt.
    """
    config = AppConfig()
    return config.model_copy(
        update={"data": config.data.model_copy(update={"source": "binance_public_data"})}
    )


def run_window(window_id: str, spec: dict[str, Any], slots: list[StrategySlot]) -> dict[str, Any]:
    """One pre-registered window: base + every mechanism on identical bars."""
    result = run_sim_live(
        bars_root=Path("data/raw/sources"),
        symbols=list(spec["symbols"]),
        interval="1d",
        config=_bench_config(),
        champion=slots[0],
        challengers=slots[1:],
        eval_tail_bars=spec["eval_tail_bars"],
        cache_dir=OUT_DIR / "qpanel_cache",
        out_dir=OUT_DIR / "receipts",
        run_id=f"chop-{window_id}",
        bench=True,
        bench_only=True,
        shared_calendar=True,
    )
    return {
        "window_id": window_id,
        "symbols": list(spec["symbols"]),
        "eval_tail_bars": spec["eval_tail_bars"],
        "receipt_run_id": result.run_id,
        "book_stats": result.receipt.get("book_stats", {}),
    }


def decision_rows(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the pre-declared decision rules to every mechanism."""
    rows = []
    for name in MECHANISMS:
        deltas = {
            wid: _delta(case["book_stats"], name)
            for wid, case in cells.items()
        }
        helped = [w for w, d in deltas.items() if d is not None and d > 0]
        hurt = [w for w, d in deltas.items() if d is not None and d < 0]
        known = [d for d in deltas.values() if d is not None]
        regime_fitting = bool(known) and bool(helped and hurt)
        w1, w3 = deltas.get("W1_full"), deltas.get("W3_last500d")
        success = (
            w3 is not None
            and w1 is not None
            and w3 > 0
            and w1 >= 0
            and not regime_fitting
            and len(known) == len(deltas)
        )
        rows.append(
            {
                "mechanism": name,
                "sharpe_deltas_vs_base": deltas,
                "helped_windows": helped,
                "hurt_windows": hurt,
                "regime_fitting": regime_fitting,
                "chop_lift_success": success,
            }
        )
    return rows


def _delta(book_stats: dict[str, Any], name: str) -> float | None:
    mechanism, base = _sharpe(book_stats.get(name, {})), _sharpe(book_stats.get("base", {}))
    if mechanism is None or base is None:
        return None
    return mechanism - base


def _sharpe(stats: dict[str, Any]) -> float | None:
    value = stats.get("sharpe_simulated")
    if not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slots = _slots()
    cells = {wid: run_window(wid, spec, slots) for wid, spec in WINDOWS.items()}
    verdicts = decision_rows(cells)
    report = {
        "kind": "chop_robustness_sweep",
        "pre_registration": "docs/SIM_LIVE_PNL.md chop-regime robustness sweep block,"
        " dated 2026-10-07T17:05+05:30",
        "run_at": "2026-10-07 (Asia/Calcutta), after the pre-registration date",
        "live_pnl_claim": False,
        "simulated_only": True,
        "research_only": True,
        "analytics_only": True,
        "data_label": "Binance USDT spot majors only; not market evidence; not a live-trading claim",
        "spec": SPEC,
        "base_policy": BASE_POLICY,
        "mechanisms": MECHANISMS,
        "windows": {k: {"symbols": v["symbols"], "eval_tail_bars": v["eval_tail_bars"]} for k, v in WINDOWS.items()},
        "cells": cells,
        "verdicts": verdicts,
    }
    (OUT_DIR / "sweep_results.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"verdicts": verdicts}, indent=2))
    for wid, case in cells.items():
        print(wid, json.dumps({k: v.get("sharpe_simulated") for k, v in case["book_stats"].items()}))


if __name__ == "__main__":
    main()
