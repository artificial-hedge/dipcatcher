"""Wave-1403 bench adapters: vision-doc-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    ai2d_lite_studies,
    chart_qa_lite_studies,
    docvqa_lite_studies,
    infovqa_lite_studies,
    mmqa_lite_studies,
    ocrvqa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14030


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ai2d_lite_studies_family(seed: int = _SEED + 0):
    """ai2d_lite_studies: synthetic correctness bench."""
    return _finite_blob(ai2d_lite_studies.bench_ai2d_lite_studies(seed))


def bench_chart_qa_lite_studies_family(seed: int = _SEED + 1):
    """chart_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(chart_qa_lite_studies.bench_chart_qa_lite_studies(seed))


def bench_docvqa_lite_studies_family(seed: int = _SEED + 2):
    """docvqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(docvqa_lite_studies.bench_docvqa_lite_studies(seed))


def bench_infovqa_lite_studies_family(seed: int = _SEED + 3):
    """infovqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(infovqa_lite_studies.bench_infovqa_lite_studies(seed))


def bench_mmqa_lite_studies_family(seed: int = _SEED + 4):
    """mmqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(mmqa_lite_studies.bench_mmqa_lite_studies(seed))


def bench_ocrvqa_lite_studies_family(seed: int = _SEED + 5):
    """ocrvqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(ocrvqa_lite_studies.bench_ocrvqa_lite_studies(seed))
