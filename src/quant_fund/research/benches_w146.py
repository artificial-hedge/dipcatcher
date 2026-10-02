"""Wave-126 adapters: exec-summary test-time-compute + multi-agent canon — consistency_vote,
verifier_prm, mcts_reason, debate_multiagent, unlearn_ga, knowledge_graph_embed —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.consistency_vote import bench_consistency_vote
from quant_fund.models.debate_multiagent import bench_debate_multiagent
from quant_fund.models.knowledge_graph_embed import bench_knowledge_graph_embed
from quant_fund.models.mcts_reason import bench_mcts_reason
from quant_fund.models.unlearn_ga import bench_unlearn_ga
from quant_fund.models.verifier_prm import bench_verifier_prm

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


def bench_consistency_vote_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("consistency_vote", bench_consistency_vote(seed=_SEED + 864)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"consistency_vote bench failed: {exc}") from exc


def bench_verifier_prm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("verifier_prm", bench_verifier_prm(seed=_SEED + 865)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"verifier_prm bench failed: {exc}") from exc


def bench_mcts_reason_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mcts_reason", bench_mcts_reason(seed=_SEED + 866)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mcts_reason bench failed: {exc}") from exc


def bench_debate_multiagent_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("debate_multiagent", bench_debate_multiagent(seed=_SEED + 867)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"debate_multiagent bench failed: {exc}") from exc


def bench_unlearn_ga_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("unlearn_ga", bench_unlearn_ga(seed=_SEED + 868)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"unlearn_ga bench failed: {exc}") from exc


def bench_knowledge_graph_embed_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("knowledge_graph_embed", bench_knowledge_graph_embed(seed=_SEED + 869))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"knowledge_graph_embed bench failed: {exc}") from exc
