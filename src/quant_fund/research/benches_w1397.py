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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
