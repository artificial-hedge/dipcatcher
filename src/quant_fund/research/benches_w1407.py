"""Wave-1407 bench adapters: conversational-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    canard_lite_studies,
    clarq_lite_studies,
    doqa_lite_studies,
    duread_qa_studies,
    orchid_qa_studies,
    qrecc_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14070


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_canard_lite_studies_family(seed: int = _SEED + 0):
    """canard_lite_studies: synthetic correctness bench."""
    return _finite_blob(canard_lite_studies.bench_canard_lite_studies(seed))


def bench_clarq_lite_studies_family(seed: int = _SEED + 1):
    """clarq_lite_studies: synthetic correctness bench."""
    return _finite_blob(clarq_lite_studies.bench_clarq_lite_studies(seed))


def bench_doqa_lite_studies_family(seed: int = _SEED + 2):
    """doqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(doqa_lite_studies.bench_doqa_lite_studies(seed))


def bench_duread_qa_studies_family(seed: int = _SEED + 3):
    """duread_qa_studies: synthetic correctness bench."""
    return _finite_blob(duread_qa_studies.bench_duread_qa_studies(seed))


def bench_orchid_qa_studies_family(seed: int = _SEED + 4):
    """orchid_qa_studies: synthetic correctness bench."""
    return _finite_blob(orchid_qa_studies.bench_orchid_qa_studies(seed))


def bench_qrecc_lite_studies_family(seed: int = _SEED + 5):
    """qrecc_lite_studies: synthetic correctness bench."""
    return _finite_blob(qrecc_lite_studies.bench_qrecc_lite_studies(seed))
