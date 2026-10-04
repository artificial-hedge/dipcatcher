"""Wave-1411 bench adapters: stance-toxicity canon (SYNTHETIC only)."""

from quant_fund.models import (
    fakeqa_lite_studies,
    flame_qa_studies,
    hate_qa_studies,
    ironic_qa_studies,
    offensive_qa_studies,
    politeness_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14110


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fakeqa_lite_studies_family(seed: int = _SEED + 0):
    """fakeqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(fakeqa_lite_studies.bench_fakeqa_lite_studies(seed))


def bench_flame_qa_studies_family(seed: int = _SEED + 1):
    """flame_qa_studies: synthetic correctness bench."""
    return _finite_blob(flame_qa_studies.bench_flame_qa_studies(seed))


def bench_hate_qa_studies_family(seed: int = _SEED + 2):
    """hate_qa_studies: synthetic correctness bench."""
    return _finite_blob(hate_qa_studies.bench_hate_qa_studies(seed))


def bench_ironic_qa_studies_family(seed: int = _SEED + 3):
    """ironic_qa_studies: synthetic correctness bench."""
    return _finite_blob(ironic_qa_studies.bench_ironic_qa_studies(seed))


def bench_offensive_qa_studies_family(seed: int = _SEED + 4):
    """offensive_qa_studies: synthetic correctness bench."""
    return _finite_blob(offensive_qa_studies.bench_offensive_qa_studies(seed))


def bench_politeness_qa_studies_family(seed: int = _SEED + 5):
    """politeness_qa_studies: synthetic correctness bench."""
    return _finite_blob(politeness_qa_studies.bench_politeness_qa_studies(seed))
