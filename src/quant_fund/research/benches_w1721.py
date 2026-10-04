"""Wave-1721 bench adapters: tatar-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    albasti_qa_studies,
    erlik_qa_studies,
    shurale_qa_studies,
    suana_qa_studies,
    tengri_qa_studies,
    umai_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17210


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_albasti_qa_studies_family(seed: int = _SEED + 0):
    """albasti_qa_studies: synthetic correctness bench."""
    return _finite_blob(albasti_qa_studies.bench_albasti_qa_studies(seed))


def bench_erlik_qa_studies_family(seed: int = _SEED + 1):
    """erlik_qa_studies: synthetic correctness bench."""
    return _finite_blob(erlik_qa_studies.bench_erlik_qa_studies(seed))


def bench_shurale_qa_studies_family(seed: int = _SEED + 2):
    """shurale_qa_studies: synthetic correctness bench."""
    return _finite_blob(shurale_qa_studies.bench_shurale_qa_studies(seed))


def bench_suana_qa_studies_family(seed: int = _SEED + 3):
    """suana_qa_studies: synthetic correctness bench."""
    return _finite_blob(suana_qa_studies.bench_suana_qa_studies(seed))


def bench_tengri_qa_studies_family(seed: int = _SEED + 4):
    """tengri_qa_studies: synthetic correctness bench."""
    return _finite_blob(tengri_qa_studies.bench_tengri_qa_studies(seed))


def bench_umai_qa_studies_family(seed: int = _SEED + 5):
    """umai_qa_studies: synthetic correctness bench."""
    return _finite_blob(umai_qa_studies.bench_umai_qa_studies(seed))
