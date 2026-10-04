"""Wave-1766 bench adapters: mayan-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cabrakan_qa_studies,
    camazotz_qa_studies,
    hunab_qa_studies,
    itzamna_qa_studies,
    ixmucane_qa_studies,
    zipacna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17660


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cabrakan_qa_studies_family(seed: int = _SEED + 0):
    """cabrakan_qa_studies: synthetic correctness bench."""
    return _finite_blob(cabrakan_qa_studies.bench_cabrakan_qa_studies(seed))


def bench_camazotz_qa_studies_family(seed: int = _SEED + 1):
    """camazotz_qa_studies: synthetic correctness bench."""
    return _finite_blob(camazotz_qa_studies.bench_camazotz_qa_studies(seed))


def bench_hunab_qa_studies_family(seed: int = _SEED + 2):
    """hunab_qa_studies: synthetic correctness bench."""
    return _finite_blob(hunab_qa_studies.bench_hunab_qa_studies(seed))


def bench_itzamna_qa_studies_family(seed: int = _SEED + 3):
    """itzamna_qa_studies: synthetic correctness bench."""
    return _finite_blob(itzamna_qa_studies.bench_itzamna_qa_studies(seed))


def bench_ixmucane_qa_studies_family(seed: int = _SEED + 4):
    """ixmucane_qa_studies: synthetic correctness bench."""
    return _finite_blob(ixmucane_qa_studies.bench_ixmucane_qa_studies(seed))


def bench_zipacna_qa_studies_family(seed: int = _SEED + 5):
    """zipacna_qa_studies: synthetic correctness bench."""
    return _finite_blob(zipacna_qa_studies.bench_zipacna_qa_studies(seed))
