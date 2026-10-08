"""Wave-1277 bench adapters: scalable-oversight canon (SYNTHETIC only)."""

from quant_fund.models import (
    debate_alignment_studies,
    deliberative_alignment_studies,
    iterated_amplification_studies,
    recursive_reward_studies,
    scalable_oversight_studies,
    weak_to_strong_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12770


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_debate_alignment_studies_family(seed: int = _SEED + 0):
    """debate_alignment_studies: synthetic correctness bench."""
    return _finite_blob(debate_alignment_studies.bench_debate_alignment_studies(seed))


def bench_deliberative_alignment_studies_family(seed: int = _SEED + 1):
    """deliberative_alignment_studies: synthetic correctness bench."""
    return _finite_blob(deliberative_alignment_studies.bench_deliberative_alignment_studies(seed))


def bench_iterated_amplification_studies_family(seed: int = _SEED + 2):
    """iterated_amplification_studies: synthetic correctness bench."""
    return _finite_blob(iterated_amplification_studies.bench_iterated_amplification_studies(seed))


def bench_recursive_reward_studies_family(seed: int = _SEED + 3):
    """recursive_reward_studies: synthetic correctness bench."""
    return _finite_blob(recursive_reward_studies.bench_recursive_reward_studies(seed))


def bench_scalable_oversight_studies_family(seed: int = _SEED + 4):
    """scalable_oversight_studies: synthetic correctness bench."""
    return _finite_blob(scalable_oversight_studies.bench_scalable_oversight_studies(seed))


def bench_weak_to_strong_studies_family(seed: int = _SEED + 5):
    """weak_to_strong_studies: synthetic correctness bench."""
    return _finite_blob(weak_to_strong_studies.bench_weak_to_strong_studies(seed))
