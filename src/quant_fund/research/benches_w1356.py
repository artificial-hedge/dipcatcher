"""Wave-1356 bench adapters: knowledge-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    bigbench_lite_studies,
    entity_qa_studies,
    mmlu_lite_studies,
    natural_qa_studies,
    pop_qa_studies,
    triviaqa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bigbench_lite_studies_family(seed: int = _SEED + 0):
    """bigbench_lite_studies: synthetic correctness bench."""
    return _finite_blob(bigbench_lite_studies.bench_bigbench_lite_studies(seed))


def bench_entity_qa_studies_family(seed: int = _SEED + 1):
    """entity_qa_studies: synthetic correctness bench."""
    return _finite_blob(entity_qa_studies.bench_entity_qa_studies(seed))


def bench_mmlu_lite_studies_family(seed: int = _SEED + 2):
    """mmlu_lite_studies: synthetic correctness bench."""
    return _finite_blob(mmlu_lite_studies.bench_mmlu_lite_studies(seed))


def bench_natural_qa_studies_family(seed: int = _SEED + 3):
    """natural_qa_studies: synthetic correctness bench."""
    return _finite_blob(natural_qa_studies.bench_natural_qa_studies(seed))


def bench_pop_qa_studies_family(seed: int = _SEED + 4):
    """pop_qa_studies: synthetic correctness bench."""
    return _finite_blob(pop_qa_studies.bench_pop_qa_studies(seed))


def bench_triviaqa_lite_studies_family(seed: int = _SEED + 5):
    """triviaqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(triviaqa_lite_studies.bench_triviaqa_lite_studies(seed))
