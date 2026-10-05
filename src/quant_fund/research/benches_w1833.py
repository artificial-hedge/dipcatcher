"""Wave-1833 bench adapters: armenian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    anahit2_qa_studies,
    aramazd2_qa_studies,
    astghik2_qa_studies,
    mher2_qa_studies,
    tir2_qa_studies,
    vahagn2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18330


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_anahit2_qa_studies_family(seed: int = _SEED + 0):
    """anahit2_qa_studies: synthetic correctness bench."""
    return _finite_blob(anahit2_qa_studies.bench_anahit2_qa_studies(seed))


def bench_aramazd2_qa_studies_family(seed: int = _SEED + 1):
    """aramazd2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aramazd2_qa_studies.bench_aramazd2_qa_studies(seed))


def bench_astghik2_qa_studies_family(seed: int = _SEED + 2):
    """astghik2_qa_studies: synthetic correctness bench."""
    return _finite_blob(astghik2_qa_studies.bench_astghik2_qa_studies(seed))


def bench_mher2_qa_studies_family(seed: int = _SEED + 3):
    """mher2_qa_studies: synthetic correctness bench."""
    return _finite_blob(mher2_qa_studies.bench_mher2_qa_studies(seed))


def bench_tir2_qa_studies_family(seed: int = _SEED + 4):
    """tir2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tir2_qa_studies.bench_tir2_qa_studies(seed))


def bench_vahagn2_qa_studies_family(seed: int = _SEED + 5):
    """vahagn2_qa_studies: synthetic correctness bench."""
    return _finite_blob(vahagn2_qa_studies.bench_vahagn2_qa_studies(seed))
