"""Wave-126 adapters: exec-summary meta-learning canon — reptile,
protonet, matching_net, anil_meta, meta_sgd, r2d2_meta —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anil_meta import bench_anil_meta
from quant_fund.models.matching_net import bench_matching_net
from quant_fund.models.meta_sgd import bench_meta_sgd
from quant_fund.models.protonet import bench_protonet
from quant_fund.models.r2d2_meta import bench_r2d2_meta
from quant_fund.models.reptile import bench_reptile

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


def bench_reptile_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("reptile", bench_reptile(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reptile bench failed: {exc}") from exc


def bench_protonet_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("protonet", bench_protonet(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"protonet bench failed: {exc}") from exc


def bench_matching_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("matching_net", bench_matching_net(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"matching_net bench failed: {exc}") from exc


def bench_anil_meta_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("anil_meta", bench_anil_meta(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"anil_meta bench failed: {exc}") from exc


def bench_meta_sgd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("meta_sgd", bench_meta_sgd(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"meta_sgd bench failed: {exc}") from exc


def bench_r2d2_meta_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("r2d2_meta", bench_r2d2_meta(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"r2d2_meta bench failed: {exc}") from exc
