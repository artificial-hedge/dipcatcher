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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
