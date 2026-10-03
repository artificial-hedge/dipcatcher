"""Wave-126 adapters: exec-summary tabular-DL canon — tokenizer_bpe,
tabular_resnet, node_net, grownet_boost, soft_tree, tabm_mini —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.grownet_boost import bench_grownet_boost
from quant_fund.models.node_net import bench_node_net
from quant_fund.models.soft_tree import bench_soft_tree
from quant_fund.models.tabm_mini import bench_tabm_mini
from quant_fund.models.tabular_resnet import bench_tabular_resnet
from quant_fund.models.tokenizer_bpe import bench_tokenizer_bpe

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


def bench_tokenizer_bpe_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tokenizer_bpe", bench_tokenizer_bpe(seed=_SEED + 924)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tokenizer_bpe bench failed: {exc}") from exc


def bench_tabular_resnet_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tabular_resnet", bench_tabular_resnet(seed=_SEED + 925)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tabular_resnet bench failed: {exc}") from exc


def bench_node_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("node_net", bench_node_net(seed=_SEED + 926)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"node_net bench failed: {exc}") from exc


def bench_grownet_boost_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("grownet_boost", bench_grownet_boost(seed=_SEED + 927)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"grownet_boost bench failed: {exc}") from exc


def bench_soft_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("soft_tree", bench_soft_tree(seed=_SEED + 928)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"soft_tree bench failed: {exc}") from exc


def bench_tabm_mini_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tabm_mini", bench_tabm_mini(seed=_SEED + 929)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tabm_mini bench failed: {exc}") from exc
