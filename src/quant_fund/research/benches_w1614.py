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
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
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
