"""Wave-126 adapters: exec-summary data-dynamics canon — dataset_distillation,
coreset_herding, curriculum_magnitude, label_smoothing, mixup_cutmix, sharpness_sam —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coreset_herding import bench_coreset_herding
from quant_fund.models.curriculum_magnitude import bench_curriculum_magnitude
from quant_fund.models.dataset_distillation import bench_dataset_distillation
from quant_fund.models.label_smoothing import bench_label_smoothing
from quant_fund.models.mixup_cutmix import bench_mixup_cutmix
from quant_fund.models.sharpness_sam import bench_sharpness_sam

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


def bench_dataset_distillation_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("dataset_distillation", bench_dataset_distillation(seed=_SEED + 882))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dataset_distillation bench failed: {exc}") from exc


def bench_coreset_herding_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("coreset_herding", bench_coreset_herding(seed=_SEED + 883)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"coreset_herding bench failed: {exc}") from exc


def bench_curriculum_magnitude_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("curriculum_magnitude", bench_curriculum_magnitude(seed=_SEED + 884))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"curriculum_magnitude bench failed: {exc}") from exc


def bench_label_smoothing_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("label_smoothing", bench_label_smoothing(seed=_SEED + 885)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"label_smoothing bench failed: {exc}") from exc


def bench_mixup_cutmix_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mixup_cutmix", bench_mixup_cutmix(seed=_SEED + 886)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mixup_cutmix bench failed: {exc}") from exc


def bench_sharpness_sam_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sharpness_sam", bench_sharpness_sam(seed=_SEED + 887)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sharpness_sam bench failed: {exc}") from exc
