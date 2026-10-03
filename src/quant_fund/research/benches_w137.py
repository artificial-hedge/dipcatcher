"""Wave-126 adapters: exec-summary distributional-RL canon — c51_dqn,
qr_dqn, iqn_dqn, noisy_net, prioritized_replay, bootstrapped_dqn —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bootstrapped_dqn import bench_bootstrapped_dqn
from quant_fund.models.c51_dqn import bench_c51_dqn
from quant_fund.models.iqn_dqn import bench_iqn_dqn
from quant_fund.models.noisy_net import bench_noisy_net
from quant_fund.models.prioritized_replay import bench_prioritized_replay
from quant_fund.models.qr_dqn import bench_qr_dqn

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


def bench_c51_dqn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("c51_dqn", bench_c51_dqn(seed=_SEED + 810)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"c51_dqn bench failed: {exc}") from exc


def bench_qr_dqn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qr_dqn", bench_qr_dqn(seed=_SEED + 811)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qr_dqn bench failed: {exc}") from exc


def bench_iqn_dqn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("iqn_dqn", bench_iqn_dqn(seed=_SEED + 812)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"iqn_dqn bench failed: {exc}") from exc


def bench_noisy_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("noisy_net", bench_noisy_net(seed=_SEED + 813)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"noisy_net bench failed: {exc}") from exc


def bench_prioritized_replay_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("prioritized_replay", bench_prioritized_replay(seed=_SEED + 814))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"prioritized_replay bench failed: {exc}") from exc


def bench_bootstrapped_dqn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bootstrapped_dqn", bench_bootstrapped_dqn(seed=_SEED + 815)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bootstrapped_dqn bench failed: {exc}") from exc
