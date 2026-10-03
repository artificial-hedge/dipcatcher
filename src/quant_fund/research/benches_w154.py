"""Wave-126 adapters: exec-summary vision canon — convnet_baseline,
vit_classifier, clip_align, simclr_views, diffusion_ddim, attention_rollout —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.attention_rollout import bench_attention_rollout
from quant_fund.models.clip_align import bench_clip_align
from quant_fund.models.convnet_baseline import bench_convnet_baseline
from quant_fund.models.diffusion_ddim import bench_diffusion_ddim
from quant_fund.models.simclr_views import bench_simclr_views
from quant_fund.models.vit_classifier import bench_vit_classifier

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


def bench_convnet_baseline_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("convnet_baseline", bench_convnet_baseline(seed=_SEED + 912)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"convnet_baseline bench failed: {exc}") from exc


def bench_vit_classifier_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vit_classifier", bench_vit_classifier(seed=_SEED + 913)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vit_classifier bench failed: {exc}") from exc


def bench_clip_align_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("clip_align", bench_clip_align(seed=_SEED + 914)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"clip_align bench failed: {exc}") from exc


def bench_simclr_views_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("simclr_views", bench_simclr_views(seed=_SEED + 915)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"simclr_views bench failed: {exc}") from exc


def bench_diffusion_ddim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("diffusion_ddim", bench_diffusion_ddim(seed=_SEED + 916)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"diffusion_ddim bench failed: {exc}") from exc


def bench_attention_rollout_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("attention_rollout", bench_attention_rollout(seed=_SEED + 917)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"attention_rollout bench failed: {exc}") from exc
