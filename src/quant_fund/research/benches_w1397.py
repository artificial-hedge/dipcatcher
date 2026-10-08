"""Wave-1397 bench adapters: multilingual-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    coma_qa_studies,
    gaia_lite_studies,
    simple_qa_studies,
    sqa_lite_studies,
    tqa_lite_studies,
    tydiqa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13970


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


def bench_coma_qa_studies_family(seed: int = _SEED + 0):
    """coma_qa_studies: synthetic correctness bench."""
    return _finite_blob(coma_qa_studies.bench_coma_qa_studies(seed))


def bench_gaia_lite_studies_family(seed: int = _SEED + 1):
    """gaia_lite_studies: synthetic correctness bench."""
    return _finite_blob(gaia_lite_studies.bench_gaia_lite_studies(seed))


def bench_simple_qa_studies_family(seed: int = _SEED + 2):
    """simple_qa_studies: synthetic correctness bench."""
    return _finite_blob(simple_qa_studies.bench_simple_qa_studies(seed))


def bench_sqa_lite_studies_family(seed: int = _SEED + 3):
    """sqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(sqa_lite_studies.bench_sqa_lite_studies(seed))


def bench_tqa_lite_studies_family(seed: int = _SEED + 4):
    """tqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(tqa_lite_studies.bench_tqa_lite_studies(seed))


def bench_tydiqa_lite_studies_family(seed: int = _SEED + 5):
    """tydiqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(tydiqa_lite_studies.bench_tydiqa_lite_studies(seed))
