"""Wave-1398 bench adapters: table-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    doc2dial_studies,
    finqa_lite_studies,
    hybridqa_lite_studies,
    infotabs_studies,
    ottqa_lite_studies,
    tab_cwq_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13980


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_doc2dial_studies_family(seed: int = _SEED + 0):
    """doc2dial_studies: synthetic correctness bench."""
    return _finite_blob(doc2dial_studies.bench_doc2dial_studies(seed))


def bench_finqa_lite_studies_family(seed: int = _SEED + 1):
    """finqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(finqa_lite_studies.bench_finqa_lite_studies(seed))


def bench_hybridqa_lite_studies_family(seed: int = _SEED + 2):
    """hybridqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(hybridqa_lite_studies.bench_hybridqa_lite_studies(seed))


def bench_infotabs_studies_family(seed: int = _SEED + 3):
    """infotabs_studies: synthetic correctness bench."""
    return _finite_blob(infotabs_studies.bench_infotabs_studies(seed))


def bench_ottqa_lite_studies_family(seed: int = _SEED + 4):
    """ottqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(ottqa_lite_studies.bench_ottqa_lite_studies(seed))


def bench_tab_cwq_studies_family(seed: int = _SEED + 5):
    """tab_cwq_studies: synthetic correctness bench."""
    return _finite_blob(tab_cwq_studies.bench_tab_cwq_studies(seed))
