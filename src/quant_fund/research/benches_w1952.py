"""Wave-1952 bench adapters: goetic-decree canon (SYNTHETIC only)."""

from quant_fund.models import (
    alloces_qa_studies,
    balam_qa_studies,
    camio_qa_studies,
    foras_qa_studies,
    furcas_qa_studies,
    gaap_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alloces_qa_studies_family(seed: int = _SEED + 0):
    """alloces_qa_studies: synthetic correctness bench."""
    return _finite_blob(alloces_qa_studies.bench_alloces_qa_studies(seed))


def bench_balam_qa_studies_family(seed: int = _SEED + 1):
    """balam_qa_studies: synthetic correctness bench."""
    return _finite_blob(balam_qa_studies.bench_balam_qa_studies(seed))


def bench_camio_qa_studies_family(seed: int = _SEED + 2):
    """camio_qa_studies: synthetic correctness bench."""
    return _finite_blob(camio_qa_studies.bench_camio_qa_studies(seed))


def bench_foras_qa_studies_family(seed: int = _SEED + 3):
    """foras_qa_studies: synthetic correctness bench."""
    return _finite_blob(foras_qa_studies.bench_foras_qa_studies(seed))


def bench_furcas_qa_studies_family(seed: int = _SEED + 4):
    """furcas_qa_studies: synthetic correctness bench."""
    return _finite_blob(furcas_qa_studies.bench_furcas_qa_studies(seed))


def bench_gaap_qa_studies_family(seed: int = _SEED + 5):
    """gaap_qa_studies: synthetic correctness bench."""
    return _finite_blob(gaap_qa_studies.bench_gaap_qa_studies(seed))
