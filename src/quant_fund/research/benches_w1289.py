"""Wave-1289 bench adapters: reasoning/CoT canon (SYNTHETIC only)."""

from quant_fund.models import (
    analogical_prompt_studies,
    cot_studies,
    reflexion_studies,
    scratchpad_studies,
    self_consistency_studies,
    stepwise_verify_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12890


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


def bench_analogical_prompt_studies_family(seed: int = _SEED + 0):
    """analogical_prompt_studies: synthetic correctness bench."""
    return _finite_blob(analogical_prompt_studies.bench_analogical_prompt_studies(seed))


def bench_cot_studies_family(seed: int = _SEED + 1):
    """cot_studies: synthetic correctness bench."""
    return _finite_blob(cot_studies.bench_cot_studies(seed))


def bench_reflexion_studies_family(seed: int = _SEED + 2):
    """reflexion_studies: synthetic correctness bench."""
    return _finite_blob(reflexion_studies.bench_reflexion_studies(seed))


def bench_scratchpad_studies_family(seed: int = _SEED + 3):
    """scratchpad_studies: synthetic correctness bench."""
    return _finite_blob(scratchpad_studies.bench_scratchpad_studies(seed))


def bench_self_consistency_studies_family(seed: int = _SEED + 4):
    """self_consistency_studies: synthetic correctness bench."""
    return _finite_blob(self_consistency_studies.bench_self_consistency_studies(seed))


def bench_stepwise_verify_studies_family(seed: int = _SEED + 5):
    """stepwise_verify_studies: synthetic correctness bench."""
    return _finite_blob(stepwise_verify_studies.bench_stepwise_verify_studies(seed))
