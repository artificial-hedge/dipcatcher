"""Wave-1713 bench adapters: georgian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    amirani_qa_studies,
    apsat_qa_studies,
    barbale_qa_studies,
    dalis_qa_studies,
    ghmerti_qa_studies,
    kamar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17130


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amirani_qa_studies_family(seed: int = _SEED + 0):
    """amirani_qa_studies: synthetic correctness bench."""
    return _finite_blob(amirani_qa_studies.bench_amirani_qa_studies(seed))


def bench_apsat_qa_studies_family(seed: int = _SEED + 1):
    """apsat_qa_studies: synthetic correctness bench."""
    return _finite_blob(apsat_qa_studies.bench_apsat_qa_studies(seed))


def bench_barbale_qa_studies_family(seed: int = _SEED + 2):
    """barbale_qa_studies: synthetic correctness bench."""
    return _finite_blob(barbale_qa_studies.bench_barbale_qa_studies(seed))


def bench_dalis_qa_studies_family(seed: int = _SEED + 3):
    """dalis_qa_studies: synthetic correctness bench."""
    return _finite_blob(dalis_qa_studies.bench_dalis_qa_studies(seed))


def bench_ghmerti_qa_studies_family(seed: int = _SEED + 4):
    """ghmerti_qa_studies: synthetic correctness bench."""
    return _finite_blob(ghmerti_qa_studies.bench_ghmerti_qa_studies(seed))


def bench_kamar_qa_studies_family(seed: int = _SEED + 5):
    """kamar_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamar_qa_studies.bench_kamar_qa_studies(seed))
