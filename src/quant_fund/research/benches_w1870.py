"""Wave-1870 bench adapters: cornish-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    bucca_qa_studies,
    knocker_qa_studies,
    morgawr_qa_studies,
    piskie_qa_studies,
    spriggan_qa_studies,
    tregeagle_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18700


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bucca_qa_studies_family(seed: int = _SEED + 0):
    """bucca_qa_studies: synthetic correctness bench."""
    return _finite_blob(bucca_qa_studies.bench_bucca_qa_studies(seed))


def bench_knocker_qa_studies_family(seed: int = _SEED + 1):
    """knocker_qa_studies: synthetic correctness bench."""
    return _finite_blob(knocker_qa_studies.bench_knocker_qa_studies(seed))


def bench_morgawr_qa_studies_family(seed: int = _SEED + 2):
    """morgawr_qa_studies: synthetic correctness bench."""
    return _finite_blob(morgawr_qa_studies.bench_morgawr_qa_studies(seed))


def bench_piskie_qa_studies_family(seed: int = _SEED + 3):
    """piskie_qa_studies: synthetic correctness bench."""
    return _finite_blob(piskie_qa_studies.bench_piskie_qa_studies(seed))


def bench_spriggan_qa_studies_family(seed: int = _SEED + 4):
    """spriggan_qa_studies: synthetic correctness bench."""
    return _finite_blob(spriggan_qa_studies.bench_spriggan_qa_studies(seed))


def bench_tregeagle_qa_studies_family(seed: int = _SEED + 5):
    """tregeagle_qa_studies: synthetic correctness bench."""
    return _finite_blob(tregeagle_qa_studies.bench_tregeagle_qa_studies(seed))
