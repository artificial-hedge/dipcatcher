"""Wave-1807 bench adapters: hittite-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hannahanna2_qa_studies,
    ilib2_qa_studies,
    kamrusepa2_qa_studies,
    kumarbi2_qa_studies,
    pirinkir2_qa_studies,
    tesub2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18070


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hannahanna2_qa_studies_family(seed: int = _SEED + 0):
    """hannahanna2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hannahanna2_qa_studies.bench_hannahanna2_qa_studies(seed))


def bench_ilib2_qa_studies_family(seed: int = _SEED + 1):
    """ilib2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ilib2_qa_studies.bench_ilib2_qa_studies(seed))


def bench_kamrusepa2_qa_studies_family(seed: int = _SEED + 2):
    """kamrusepa2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kamrusepa2_qa_studies.bench_kamrusepa2_qa_studies(seed))


def bench_kumarbi2_qa_studies_family(seed: int = _SEED + 3):
    """kumarbi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kumarbi2_qa_studies.bench_kumarbi2_qa_studies(seed))


def bench_pirinkir2_qa_studies_family(seed: int = _SEED + 4):
    """pirinkir2_qa_studies: synthetic correctness bench."""
    return _finite_blob(pirinkir2_qa_studies.bench_pirinkir2_qa_studies(seed))


def bench_tesub2_qa_studies_family(seed: int = _SEED + 5):
    """tesub2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tesub2_qa_studies.bench_tesub2_qa_studies(seed))
