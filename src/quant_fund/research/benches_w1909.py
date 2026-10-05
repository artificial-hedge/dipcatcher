"""Wave-1909 bench adapters: persian-daeva canon (SYNTHETIC only)."""

from quant_fund.models import (
    aeshma_qa_studies,
    astwihad_qa_studies,
    azhi_dahaka_qa_studies,
    druj_qa_studies,
    jahi_qa_studies,
    nasu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19090


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aeshma_qa_studies_family(seed: int = _SEED + 0):
    """aeshma_qa_studies: synthetic correctness bench."""
    return _finite_blob(aeshma_qa_studies.bench_aeshma_qa_studies(seed))


def bench_astwihad_qa_studies_family(seed: int = _SEED + 1):
    """astwihad_qa_studies: synthetic correctness bench."""
    return _finite_blob(astwihad_qa_studies.bench_astwihad_qa_studies(seed))


def bench_azhi_dahaka_qa_studies_family(seed: int = _SEED + 2):
    """azhi_dahaka_qa_studies: synthetic correctness bench."""
    return _finite_blob(azhi_dahaka_qa_studies.bench_azhi_dahaka_qa_studies(seed))


def bench_druj_qa_studies_family(seed: int = _SEED + 3):
    """druj_qa_studies: synthetic correctness bench."""
    return _finite_blob(druj_qa_studies.bench_druj_qa_studies(seed))


def bench_jahi_qa_studies_family(seed: int = _SEED + 4):
    """jahi_qa_studies: synthetic correctness bench."""
    return _finite_blob(jahi_qa_studies.bench_jahi_qa_studies(seed))


def bench_nasu_qa_studies_family(seed: int = _SEED + 5):
    """nasu_qa_studies: synthetic correctness bench."""
    return _finite_blob(nasu_qa_studies.bench_nasu_qa_studies(seed))
