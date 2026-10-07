"""Wave-1651 bench adapters: australian-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    awgy_qa_studies,
    kuritja_qa_studies,
    minka_qa_studies,
    papin_qa_studies,
    yara_qa_studies,
    yowie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16510


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


def bench_awgy_qa_studies_family(seed: int = _SEED + 0):
    """awgy_qa_studies: synthetic correctness bench."""
    return _finite_blob(awgy_qa_studies.bench_awgy_qa_studies(seed))


def bench_kuritja_qa_studies_family(seed: int = _SEED + 1):
    """kuritja_qa_studies: synthetic correctness bench."""
    return _finite_blob(kuritja_qa_studies.bench_kuritja_qa_studies(seed))


def bench_minka_qa_studies_family(seed: int = _SEED + 2):
    """minka_qa_studies: synthetic correctness bench."""
    return _finite_blob(minka_qa_studies.bench_minka_qa_studies(seed))


def bench_papin_qa_studies_family(seed: int = _SEED + 3):
    """papin_qa_studies: synthetic correctness bench."""
    return _finite_blob(papin_qa_studies.bench_papin_qa_studies(seed))


def bench_yara_qa_studies_family(seed: int = _SEED + 4):
    """yara_qa_studies: synthetic correctness bench."""
    return _finite_blob(yara_qa_studies.bench_yara_qa_studies(seed))


def bench_yowie_qa_studies_family(seed: int = _SEED + 5):
    """yowie_qa_studies: synthetic correctness bench."""
    return _finite_blob(yowie_qa_studies.bench_yowie_qa_studies(seed))
