"""Wave-317 image-processing canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.canny_edge import bench_canny_edge
from quant_fund.models.distance_transform import bench_distance_transform
from quant_fund.models.nlm_denoise import bench_nlm_denoise
from quant_fund.models.otsu_threshold import bench_otsu_threshold
from quant_fund.models.slic_superpixels import bench_slic_superpixels
from quant_fund.models.watershed_seg import bench_watershed_seg

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_canny_edge_family(seed: int = _SEED + 1809) -> dict[str, float]:
    return _floats(_finite_blob("canny_edge", bench_canny_edge(seed)))


def bench_otsu_threshold_family(seed: int = _SEED + 1810) -> dict[str, float]:
    return _floats(_finite_blob("otsu_threshold", bench_otsu_threshold(seed)))


def bench_watershed_seg_family(seed: int = _SEED + 1811) -> dict[str, float]:
    return _floats(_finite_blob("watershed_seg", bench_watershed_seg(seed)))


def bench_slic_superpixels_family(seed: int = _SEED + 1812) -> dict[str, float]:
    return _floats(_finite_blob("slic_superpixels", bench_slic_superpixels(seed)))


def bench_nlm_denoise_family(seed: int = _SEED + 1813) -> dict[str, float]:
    return _floats(_finite_blob("nlm_denoise", bench_nlm_denoise(seed)))


def bench_distance_transform_family(seed: int = _SEED + 1814) -> dict[str, float]:
    return _floats(_finite_blob("distance_transform", bench_distance_transform(seed)))
