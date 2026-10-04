"""Wave-1402 bench adapters: science-QA-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ai2_arc_lite_studies,
    arc_da_lite_studies,
    drug_qa_lite_studies,
    emrqa_lite_studies,
    head_qa_lite_studies,
    medmcqa_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14020


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ai2_arc_lite_studies_family(seed: int = _SEED + 0):
    """ai2_arc_lite_studies: synthetic correctness bench."""
    return _finite_blob(ai2_arc_lite_studies.bench_ai2_arc_lite_studies(seed))


def bench_arc_da_lite_studies_family(seed: int = _SEED + 1):
    """arc_da_lite_studies: synthetic correctness bench."""
    return _finite_blob(arc_da_lite_studies.bench_arc_da_lite_studies(seed))


def bench_drug_qa_lite_studies_family(seed: int = _SEED + 2):
    """drug_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(drug_qa_lite_studies.bench_drug_qa_lite_studies(seed))


def bench_emrqa_lite_studies_family(seed: int = _SEED + 3):
    """emrqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(emrqa_lite_studies.bench_emrqa_lite_studies(seed))


def bench_head_qa_lite_studies_family(seed: int = _SEED + 4):
    """head_qa_lite_studies: synthetic correctness bench."""
    return _finite_blob(head_qa_lite_studies.bench_head_qa_lite_studies(seed))


def bench_medmcqa_lite_studies_family(seed: int = _SEED + 5):
    """medmcqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(medmcqa_lite_studies.bench_medmcqa_lite_studies(seed))
