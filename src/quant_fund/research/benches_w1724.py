"""Wave-1724 bench adapters: polynesian-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    kamapuaa_qa_studies,
    kanaloa_qa_studies,
    pele_qa_studies,
    rongo_qa_studies,
    tane_qa_studies,
    tangaroa_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17240


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_kamapuaa_qa_studies_family(seed: int = _SEED + 0):
    """kamapuaa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamapuaa_qa_studies.bench_kamapuaa_qa_studies(seed))


def bench_kanaloa_qa_studies_family(seed: int = _SEED + 1):
    """kanaloa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kanaloa_qa_studies.bench_kanaloa_qa_studies(seed))


def bench_pele_qa_studies_family(seed: int = _SEED + 2):
    """pele_qa_studies: synthetic correctness bench."""
    return _finite_blob(pele_qa_studies.bench_pele_qa_studies(seed))


def bench_rongo_qa_studies_family(seed: int = _SEED + 3):
    """rongo_qa_studies: synthetic correctness bench."""
    return _finite_blob(rongo_qa_studies.bench_rongo_qa_studies(seed))


def bench_tane_qa_studies_family(seed: int = _SEED + 4):
    """tane_qa_studies: synthetic correctness bench."""
    return _finite_blob(tane_qa_studies.bench_tane_qa_studies(seed))


def bench_tangaroa_qa_studies_family(seed: int = _SEED + 5):
    """tangaroa_qa_studies: synthetic correctness bench."""
    return _finite_blob(tangaroa_qa_studies.bench_tangaroa_qa_studies(seed))
