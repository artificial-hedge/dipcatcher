"""Wave-1399 bench adapters: retrieval-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    asqa_lite_studies,
    eli5_lite_studies,
    fresh_qa_studies,
    nq_lite_studies,
    trivia_lite_studies,
    xor_tydi_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13990


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


def bench_asqa_lite_studies_family(seed: int = _SEED + 0):
    """asqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(asqa_lite_studies.bench_asqa_lite_studies(seed))


def bench_eli5_lite_studies_family(seed: int = _SEED + 1):
    """eli5_lite_studies: synthetic correctness bench."""
    return _finite_blob(eli5_lite_studies.bench_eli5_lite_studies(seed))


def bench_fresh_qa_studies_family(seed: int = _SEED + 2):
    """fresh_qa_studies: synthetic correctness bench."""
    return _finite_blob(fresh_qa_studies.bench_fresh_qa_studies(seed))


def bench_nq_lite_studies_family(seed: int = _SEED + 3):
    """nq_lite_studies: synthetic correctness bench."""
    return _finite_blob(nq_lite_studies.bench_nq_lite_studies(seed))


def bench_trivia_lite_studies_family(seed: int = _SEED + 4):
    """trivia_lite_studies: synthetic correctness bench."""
    return _finite_blob(trivia_lite_studies.bench_trivia_lite_studies(seed))


def bench_xor_tydi_studies_family(seed: int = _SEED + 5):
    """xor_tydi_studies: synthetic correctness bench."""
    return _finite_blob(xor_tydi_studies.bench_xor_tydi_studies(seed))
