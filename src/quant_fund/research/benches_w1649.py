"""Wave-1649 bench adapters: filipino-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    aswang_qa_studies,
    bakunawa_qa_studies,
    berbalang_qa_studies,
    kapre_qa_studies,
    sigbin_qa_studies,
    tikbalang_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16490


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


def bench_aswang_qa_studies_family(seed: int = _SEED + 0):
    """aswang_qa_studies: synthetic correctness bench."""
    return _finite_blob(aswang_qa_studies.bench_aswang_qa_studies(seed))


def bench_bakunawa_qa_studies_family(seed: int = _SEED + 1):
    """bakunawa_qa_studies: synthetic correctness bench."""
    return _finite_blob(bakunawa_qa_studies.bench_bakunawa_qa_studies(seed))


def bench_berbalang_qa_studies_family(seed: int = _SEED + 2):
    """berbalang_qa_studies: synthetic correctness bench."""
    return _finite_blob(berbalang_qa_studies.bench_berbalang_qa_studies(seed))


def bench_kapre_qa_studies_family(seed: int = _SEED + 3):
    """kapre_qa_studies: synthetic correctness bench."""
    return _finite_blob(kapre_qa_studies.bench_kapre_qa_studies(seed))


def bench_sigbin_qa_studies_family(seed: int = _SEED + 4):
    """sigbin_qa_studies: synthetic correctness bench."""
    return _finite_blob(sigbin_qa_studies.bench_sigbin_qa_studies(seed))


def bench_tikbalang_qa_studies_family(seed: int = _SEED + 5):
    """tikbalang_qa_studies: synthetic correctness bench."""
    return _finite_blob(tikbalang_qa_studies.bench_tikbalang_qa_studies(seed))
