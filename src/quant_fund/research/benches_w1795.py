"""Wave-1795 bench adapters: egyptian-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anubis2_qa_studies,
    isis2_qa_studies,
    khonsu2_qa_studies,
    osiris2_qa_studies,
    ra2_qa_studies,
    sobek2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17950


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anubis2_qa_studies_family(seed: int = _SEED + 0):
    """anubis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(anubis2_qa_studies.bench_anubis2_qa_studies(seed))


def bench_isis2_qa_studies_family(seed: int = _SEED + 1):
    """isis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(isis2_qa_studies.bench_isis2_qa_studies(seed))


def bench_khonsu2_qa_studies_family(seed: int = _SEED + 2):
    """khonsu2_qa_studies: synthetic correctness bench."""
    return _finite_blob(khonsu2_qa_studies.bench_khonsu2_qa_studies(seed))


def bench_osiris2_qa_studies_family(seed: int = _SEED + 3):
    """osiris2_qa_studies: synthetic correctness bench."""
    return _finite_blob(osiris2_qa_studies.bench_osiris2_qa_studies(seed))


def bench_ra2_qa_studies_family(seed: int = _SEED + 4):
    """ra2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ra2_qa_studies.bench_ra2_qa_studies(seed))


def bench_sobek2_qa_studies_family(seed: int = _SEED + 5):
    """sobek2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sobek2_qa_studies.bench_sobek2_qa_studies(seed))
