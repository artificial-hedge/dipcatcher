"""Wave-1586 bench adapters: plains-game canon (SYNTHETIC only)."""

from quant_fund.models import (
    beira_qa_studies,
    gemsbok_qa_studies,
    madoqua_qa_studies,
    oribi_qa_studies,
    reedbuck_qa_studies,
    tsessebe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15860


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


def bench_beira_qa_studies_family(seed: int = _SEED + 0):
    """beira_qa_studies: synthetic correctness bench."""
    return _finite_blob(beira_qa_studies.bench_beira_qa_studies(seed))


def bench_gemsbok_qa_studies_family(seed: int = _SEED + 1):
    """gemsbok_qa_studies: synthetic correctness bench."""
    return _finite_blob(gemsbok_qa_studies.bench_gemsbok_qa_studies(seed))


def bench_madoqua_qa_studies_family(seed: int = _SEED + 2):
    """madoqua_qa_studies: synthetic correctness bench."""
    return _finite_blob(madoqua_qa_studies.bench_madoqua_qa_studies(seed))


def bench_oribi_qa_studies_family(seed: int = _SEED + 3):
    """oribi_qa_studies: synthetic correctness bench."""
    return _finite_blob(oribi_qa_studies.bench_oribi_qa_studies(seed))


def bench_reedbuck_qa_studies_family(seed: int = _SEED + 4):
    """reedbuck_qa_studies: synthetic correctness bench."""
    return _finite_blob(reedbuck_qa_studies.bench_reedbuck_qa_studies(seed))


def bench_tsessebe_qa_studies_family(seed: int = _SEED + 5):
    """tsessebe_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsessebe_qa_studies.bench_tsessebe_qa_studies(seed))
