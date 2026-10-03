"""Wave-125 adapters: exec-summary pricing/quantum/XAI SOTA — pinn_pricing,
qubo_portfolio, xai_shap, adversarial_robust, risk_flow, causal_miner —
benched on SYNTHETIC corpora/generators. Adapters flatten to a finite
float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adversarial_robust import bench_adversarial_robust
from quant_fund.models.causal_miner import bench_causal_miner
from quant_fund.models.pinn_pricing import bench_pinn_pricing
from quant_fund.models.qubo_portfolio import bench_qubo_portfolio
from quant_fund.models.risk_flow import bench_risk_flow
from quant_fund.models.xai_shap import bench_xai_shap

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_pinn_pricing_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pinn_pricing", bench_pinn_pricing(seed=_SEED + 738)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pinn_pricing bench failed: {exc}") from exc


def bench_qubo_portfolio_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qubo_portfolio", bench_qubo_portfolio(seed=_SEED + 739)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qubo_portfolio bench failed: {exc}") from exc


def bench_xai_shap_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("xai_shap", bench_xai_shap(seed=_SEED + 740)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"xai_shap bench failed: {exc}") from exc


def bench_adversarial_robust_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("adversarial_robust", bench_adversarial_robust(seed=_SEED + 741))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adversarial_robust bench failed: {exc}") from exc


def bench_risk_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("risk_flow", bench_risk_flow(seed=_SEED + 742)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"risk_flow bench failed: {exc}") from exc


def bench_causal_miner_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("causal_miner", bench_causal_miner(seed=_SEED + 743)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"causal_miner bench failed: {exc}") from exc
