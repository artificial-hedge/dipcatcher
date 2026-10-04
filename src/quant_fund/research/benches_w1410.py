"""Wave-1410 bench adapters: event-causality canon (SYNTHETIC only)."""

from quant_fund.models import (
    causal_qa_studies,
    ecare_lite_studies,
    event2mind_lite_studies,
    event_qa_studies,
    hippo_qa_studies,
    intent_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14100


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_causal_qa_studies_family(seed: int = _SEED + 0):
    """causal_qa_studies: synthetic correctness bench."""
    return _finite_blob(causal_qa_studies.bench_causal_qa_studies(seed))


def bench_ecare_lite_studies_family(seed: int = _SEED + 1):
    """ecare_lite_studies: synthetic correctness bench."""
    return _finite_blob(ecare_lite_studies.bench_ecare_lite_studies(seed))


def bench_event2mind_lite_studies_family(seed: int = _SEED + 2):
    """event2mind_lite_studies: synthetic correctness bench."""
    return _finite_blob(event2mind_lite_studies.bench_event2mind_lite_studies(seed))


def bench_event_qa_studies_family(seed: int = _SEED + 3):
    """event_qa_studies: synthetic correctness bench."""
    return _finite_blob(event_qa_studies.bench_event_qa_studies(seed))


def bench_hippo_qa_studies_family(seed: int = _SEED + 4):
    """hippo_qa_studies: synthetic correctness bench."""
    return _finite_blob(hippo_qa_studies.bench_hippo_qa_studies(seed))


def bench_intent_qa_studies_family(seed: int = _SEED + 5):
    """intent_qa_studies: synthetic correctness bench."""
    return _finite_blob(intent_qa_studies.bench_intent_qa_studies(seed))
