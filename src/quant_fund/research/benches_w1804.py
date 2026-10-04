"""Wave-1804 bench adapters: sumerian-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    enki2_qa_studies,
    enlil2_qa_studies,
    gelal2_qa_studies,
    namtar2_qa_studies,
    ninurta2_qa_studies,
    zababa2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18040


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_enki2_qa_studies_family(seed: int = _SEED + 0):
    """enki2_qa_studies: synthetic correctness bench."""
    return _finite_blob(enki2_qa_studies.bench_enki2_qa_studies(seed))


def bench_enlil2_qa_studies_family(seed: int = _SEED + 1):
    """enlil2_qa_studies: synthetic correctness bench."""
    return _finite_blob(enlil2_qa_studies.bench_enlil2_qa_studies(seed))


def bench_gelal2_qa_studies_family(seed: int = _SEED + 2):
    """gelal2_qa_studies: synthetic correctness bench."""
    return _finite_blob(gelal2_qa_studies.bench_gelal2_qa_studies(seed))


def bench_namtar2_qa_studies_family(seed: int = _SEED + 3):
    """namtar2_qa_studies: synthetic correctness bench."""
    return _finite_blob(namtar2_qa_studies.bench_namtar2_qa_studies(seed))


def bench_ninurta2_qa_studies_family(seed: int = _SEED + 4):
    """ninurta2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninurta2_qa_studies.bench_ninurta2_qa_studies(seed))


def bench_zababa2_qa_studies_family(seed: int = _SEED + 5):
    """zababa2_qa_studies: synthetic correctness bench."""
    return _finite_blob(zababa2_qa_studies.bench_zababa2_qa_studies(seed))
