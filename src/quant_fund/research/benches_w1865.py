"""Wave-1865 bench adapters: arthurian-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    enid_qa_studies,
    geraint_qa_studies,
    gwalchmei_qa_studies,
    olwen_qa_studies,
    owen_qa_studies,
    rheged_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18650


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_enid_qa_studies_family(seed: int = _SEED + 0):
    """enid_qa_studies: synthetic correctness bench."""
    return _finite_blob(enid_qa_studies.bench_enid_qa_studies(seed))


def bench_geraint_qa_studies_family(seed: int = _SEED + 1):
    """geraint_qa_studies: synthetic correctness bench."""
    return _finite_blob(geraint_qa_studies.bench_geraint_qa_studies(seed))


def bench_gwalchmei_qa_studies_family(seed: int = _SEED + 2):
    """gwalchmei_qa_studies: synthetic correctness bench."""
    return _finite_blob(gwalchmei_qa_studies.bench_gwalchmei_qa_studies(seed))


def bench_olwen_qa_studies_family(seed: int = _SEED + 3):
    """olwen_qa_studies: synthetic correctness bench."""
    return _finite_blob(olwen_qa_studies.bench_olwen_qa_studies(seed))


def bench_owen_qa_studies_family(seed: int = _SEED + 4):
    """owen_qa_studies: synthetic correctness bench."""
    return _finite_blob(owen_qa_studies.bench_owen_qa_studies(seed))


def bench_rheged_qa_studies_family(seed: int = _SEED + 5):
    """rheged_qa_studies: synthetic correctness bench."""
    return _finite_blob(rheged_qa_studies.bench_rheged_qa_studies(seed))
