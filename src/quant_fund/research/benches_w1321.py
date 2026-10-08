"""Wave-1321 bench adapters: multimodal-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    chart_gqa_studies,
    mathvista_studies,
    mkqa_studies,
    mmmlu_studies,
    mmmu_studies,
    videomme_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13210


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


def bench_chart_gqa_studies_family(seed: int = _SEED + 0):
    """chart_gqa_studies: synthetic correctness bench."""
    return _finite_blob(chart_gqa_studies.bench_chart_gqa_studies(seed))


def bench_mathvista_studies_family(seed: int = _SEED + 1):
    """mathvista_studies: synthetic correctness bench."""
    return _finite_blob(mathvista_studies.bench_mathvista_studies(seed))


def bench_mkqa_studies_family(seed: int = _SEED + 2):
    """mkqa_studies: synthetic correctness bench."""
    return _finite_blob(mkqa_studies.bench_mkqa_studies(seed))


def bench_mmmlu_studies_family(seed: int = _SEED + 3):
    """mmmlu_studies: synthetic correctness bench."""
    return _finite_blob(mmmlu_studies.bench_mmmlu_studies(seed))


def bench_mmmu_studies_family(seed: int = _SEED + 4):
    """mmmu_studies: synthetic correctness bench."""
    return _finite_blob(mmmu_studies.bench_mmmu_studies(seed))


def bench_videomme_studies_family(seed: int = _SEED + 5):
    """videomme_studies: synthetic correctness bench."""
    return _finite_blob(videomme_studies.bench_videomme_studies(seed))
