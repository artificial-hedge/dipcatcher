"""Wave-1420 bench adapters: reasoning-exotics canon (SYNTHETIC only)."""

from quant_fund.models import (
    cause_qa_studies,
    claim_qa_studies,
    conclusion_qa_studies,
    deduction_qa_studies,
    effect_qa_studies,
    fallacy_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14200


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


def bench_cause_qa_studies_family(seed: int = _SEED + 0):
    """cause_qa_studies: synthetic correctness bench."""
    return _finite_blob(cause_qa_studies.bench_cause_qa_studies(seed))


def bench_claim_qa_studies_family(seed: int = _SEED + 1):
    """claim_qa_studies: synthetic correctness bench."""
    return _finite_blob(claim_qa_studies.bench_claim_qa_studies(seed))


def bench_conclusion_qa_studies_family(seed: int = _SEED + 2):
    """conclusion_qa_studies: synthetic correctness bench."""
    return _finite_blob(conclusion_qa_studies.bench_conclusion_qa_studies(seed))


def bench_deduction_qa_studies_family(seed: int = _SEED + 3):
    """deduction_qa_studies: synthetic correctness bench."""
    return _finite_blob(deduction_qa_studies.bench_deduction_qa_studies(seed))


def bench_effect_qa_studies_family(seed: int = _SEED + 4):
    """effect_qa_studies: synthetic correctness bench."""
    return _finite_blob(effect_qa_studies.bench_effect_qa_studies(seed))


def bench_fallacy_qa_studies_family(seed: int = _SEED + 5):
    """fallacy_qa_studies: synthetic correctness bench."""
    return _finite_blob(fallacy_qa_studies.bench_fallacy_qa_studies(seed))
