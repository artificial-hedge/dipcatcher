"""Wave-1527 bench adapters: lichen canon (SYNTHETIC only)."""

from quant_fund.models import (
    crustose_qa_studies,
    foliose_qa_studies,
    fruticose_qa_studies,
    oakmoss_qa_studies,
    usnea_qa_studies,
    xanthoria_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15270


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_crustose_qa_studies_family(seed: int = _SEED + 0):
    """crustose_qa_studies: synthetic correctness bench."""
    return _finite_blob(crustose_qa_studies.bench_crustose_qa_studies(seed))


def bench_foliose_qa_studies_family(seed: int = _SEED + 1):
    """foliose_qa_studies: synthetic correctness bench."""
    return _finite_blob(foliose_qa_studies.bench_foliose_qa_studies(seed))


def bench_fruticose_qa_studies_family(seed: int = _SEED + 2):
    """fruticose_qa_studies: synthetic correctness bench."""
    return _finite_blob(fruticose_qa_studies.bench_fruticose_qa_studies(seed))


def bench_oakmoss_qa_studies_family(seed: int = _SEED + 3):
    """oakmoss_qa_studies: synthetic correctness bench."""
    return _finite_blob(oakmoss_qa_studies.bench_oakmoss_qa_studies(seed))


def bench_usnea_qa_studies_family(seed: int = _SEED + 4):
    """usnea_qa_studies: synthetic correctness bench."""
    return _finite_blob(usnea_qa_studies.bench_usnea_qa_studies(seed))


def bench_xanthoria_qa_studies_family(seed: int = _SEED + 5):
    """xanthoria_qa_studies: synthetic correctness bench."""
    return _finite_blob(xanthoria_qa_studies.bench_xanthoria_qa_studies(seed))
