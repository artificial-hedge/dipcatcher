"""Wave-1708 bench adapters: nenets-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    metsik_qa_studies,
    naveluz_qa_studies,
    numishi_qa_studies,
    numit_qa_studies,
    piryani_qa_studies,
    yejmun_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17080


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


def bench_metsik_qa_studies_family(seed: int = _SEED + 0):
    """metsik_qa_studies: synthetic correctness bench."""
    return _finite_blob(metsik_qa_studies.bench_metsik_qa_studies(seed))


def bench_naveluz_qa_studies_family(seed: int = _SEED + 1):
    """naveluz_qa_studies: synthetic correctness bench."""
    return _finite_blob(naveluz_qa_studies.bench_naveluz_qa_studies(seed))


def bench_numishi_qa_studies_family(seed: int = _SEED + 2):
    """numishi_qa_studies: synthetic correctness bench."""
    return _finite_blob(numishi_qa_studies.bench_numishi_qa_studies(seed))


def bench_numit_qa_studies_family(seed: int = _SEED + 3):
    """numit_qa_studies: synthetic correctness bench."""
    return _finite_blob(numit_qa_studies.bench_numit_qa_studies(seed))


def bench_piryani_qa_studies_family(seed: int = _SEED + 4):
    """piryani_qa_studies: synthetic correctness bench."""
    return _finite_blob(piryani_qa_studies.bench_piryani_qa_studies(seed))


def bench_yejmun_qa_studies_family(seed: int = _SEED + 5):
    """yejmun_qa_studies: synthetic correctness bench."""
    return _finite_blob(yejmun_qa_studies.bench_yejmun_qa_studies(seed))
