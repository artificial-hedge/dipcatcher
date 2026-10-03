"""Wave-124 adapters: exec-summary graph/meta/continual SOTA — asset_gnn,
counterparty_gnn, maml_portfolio, continual_learning, fed_avg,
insider_anomaly — benched on SYNTHETIC corpora/generators. Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.asset_gnn import bench_asset_gnn
from quant_fund.models.continual_learning import bench_continual_learning
from quant_fund.models.counterparty_gnn import bench_counterparty_gnn
from quant_fund.models.fed_avg import bench_fed_avg
from quant_fund.models.insider_anomaly import bench_insider_anomaly
from quant_fund.models.maml_portfolio import bench_maml_portfolio

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


def bench_asset_gnn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("asset_gnn", bench_asset_gnn(seed=_SEED + 732)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"asset_gnn bench failed: {exc}") from exc


def bench_counterparty_gnn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("counterparty_gnn", bench_counterparty_gnn(seed=_SEED + 733)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"counterparty_gnn bench failed: {exc}") from exc


def bench_maml_portfolio_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("maml_portfolio", bench_maml_portfolio(seed=_SEED + 734)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"maml_portfolio bench failed: {exc}") from exc


def bench_continual_learning_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("continual_learning", bench_continual_learning(seed=_SEED + 735))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"continual_learning bench failed: {exc}") from exc


def bench_fed_avg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fed_avg", bench_fed_avg(seed=_SEED + 736)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fed_avg bench failed: {exc}") from exc


def bench_insider_anomaly_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("insider_anomaly", bench_insider_anomaly(seed=_SEED + 737)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"insider_anomaly bench failed: {exc}") from exc
