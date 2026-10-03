"""Wave-123 adapters: exec-summary NLP/generative SOTA — say_echo_do,
multimodal_fusion, ts_diffusion, synthetic_gan, econ_calendar,
quantcode_bench — benched on SYNTHETIC corpora/generators. Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.econ_calendar import bench_econ_calendar
from quant_fund.models.multimodal_fusion import bench_multimodal_fusion
from quant_fund.models.quantcode_bench import bench_quantcode_bench
from quant_fund.models.say_echo_do import bench_say_echo_do
from quant_fund.models.synthetic_gan import bench_synthetic_gan
from quant_fund.models.ts_diffusion import bench_ts_diffusion

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


def bench_say_echo_do_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("say_echo_do", bench_say_echo_do(seed=_SEED + 726)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"say_echo_do bench failed: {exc}") from exc


def bench_multimodal_fusion_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("multimodal_fusion", bench_multimodal_fusion(seed=_SEED + 727)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"multimodal_fusion bench failed: {exc}") from exc


def bench_ts_diffusion_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ts_diffusion", bench_ts_diffusion(seed=_SEED + 728)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ts_diffusion bench failed: {exc}") from exc


def bench_synthetic_gan_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("synthetic_gan", bench_synthetic_gan(seed=_SEED + 729)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"synthetic_gan bench failed: {exc}") from exc


def bench_econ_calendar_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("econ_calendar", bench_econ_calendar(seed=_SEED + 730)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"econ_calendar bench failed: {exc}") from exc


def bench_quantcode_bench_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("quantcode_bench", bench_quantcode_bench(seed=_SEED + 731)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"quantcode_bench bench failed: {exc}") from exc
