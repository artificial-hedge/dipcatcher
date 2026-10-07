"""Wave-1650 bench adapters: african-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    adjule_qa_studies,
    agogwe_qa_studies,
    biloko_qa_studies,
    kongamato_qa_studies,
    popobawa_qa_studies,
    rompo_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16500


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


def bench_adjule_qa_studies_family(seed: int = _SEED + 0):
    """adjule_qa_studies: synthetic correctness bench."""
    return _finite_blob(adjule_qa_studies.bench_adjule_qa_studies(seed))


def bench_agogwe_qa_studies_family(seed: int = _SEED + 1):
    """agogwe_qa_studies: synthetic correctness bench."""
    return _finite_blob(agogwe_qa_studies.bench_agogwe_qa_studies(seed))


def bench_biloko_qa_studies_family(seed: int = _SEED + 2):
    """biloko_qa_studies: synthetic correctness bench."""
    return _finite_blob(biloko_qa_studies.bench_biloko_qa_studies(seed))


def bench_kongamato_qa_studies_family(seed: int = _SEED + 3):
    """kongamato_qa_studies: synthetic correctness bench."""
    return _finite_blob(kongamato_qa_studies.bench_kongamato_qa_studies(seed))


def bench_popobawa_qa_studies_family(seed: int = _SEED + 4):
    """popobawa_qa_studies: synthetic correctness bench."""
    return _finite_blob(popobawa_qa_studies.bench_popobawa_qa_studies(seed))


def bench_rompo_qa_studies_family(seed: int = _SEED + 5):
    """rompo_qa_studies: synthetic correctness bench."""
    return _finite_blob(rompo_qa_studies.bench_rompo_qa_studies(seed))
