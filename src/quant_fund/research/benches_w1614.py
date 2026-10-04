"""Wave-1614 bench adapters: deer-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    hog_deer_qa_studies,
    kouprey_qa_studies,
    mule_qa_studies,
    pere_david_qa_studies,
    red_deer_qa_studies,
    wapiti_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16140


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_hog_deer_qa_studies_family(seed: int = _SEED + 0):
    """hog_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(hog_deer_qa_studies.bench_hog_deer_qa_studies(seed))


def bench_kouprey_qa_studies_family(seed: int = _SEED + 1):
    """kouprey_qa_studies: synthetic correctness bench."""
    return _finite_blob(kouprey_qa_studies.bench_kouprey_qa_studies(seed))


def bench_mule_qa_studies_family(seed: int = _SEED + 2):
    """mule_qa_studies: synthetic correctness bench."""
    return _finite_blob(mule_qa_studies.bench_mule_qa_studies(seed))


def bench_pere_david_qa_studies_family(seed: int = _SEED + 3):
    """pere_david_qa_studies: synthetic correctness bench."""
    return _finite_blob(pere_david_qa_studies.bench_pere_david_qa_studies(seed))


def bench_red_deer_qa_studies_family(seed: int = _SEED + 4):
    """red_deer_qa_studies: synthetic correctness bench."""
    return _finite_blob(red_deer_qa_studies.bench_red_deer_qa_studies(seed))


def bench_wapiti_qa_studies_family(seed: int = _SEED + 5):
    """wapiti_qa_studies: synthetic correctness bench."""
    return _finite_blob(wapiti_qa_studies.bench_wapiti_qa_studies(seed))
