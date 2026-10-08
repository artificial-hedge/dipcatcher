"""Wave-1417 bench adapters: instruction-task canon (SYNTHETIC only)."""

from quant_fund.models import (
    checklist_qa_studies,
    flow_qa_studies,
    guide_qa_studies,
    howto_qa_studies,
    instruct_qa_studies,
    lesson_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14170


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_checklist_qa_studies_family(seed: int = _SEED + 0):
    """checklist_qa_studies: synthetic correctness bench."""
    return _finite_blob(checklist_qa_studies.bench_checklist_qa_studies(seed))


def bench_flow_qa_studies_family(seed: int = _SEED + 1):
    """flow_qa_studies: synthetic correctness bench."""
    return _finite_blob(flow_qa_studies.bench_flow_qa_studies(seed))


def bench_guide_qa_studies_family(seed: int = _SEED + 2):
    """guide_qa_studies: synthetic correctness bench."""
    return _finite_blob(guide_qa_studies.bench_guide_qa_studies(seed))


def bench_howto_qa_studies_family(seed: int = _SEED + 3):
    """howto_qa_studies: synthetic correctness bench."""
    return _finite_blob(howto_qa_studies.bench_howto_qa_studies(seed))


def bench_instruct_qa_studies_family(seed: int = _SEED + 4):
    """instruct_qa_studies: synthetic correctness bench."""
    return _finite_blob(instruct_qa_studies.bench_instruct_qa_studies(seed))


def bench_lesson_qa_studies_family(seed: int = _SEED + 5):
    """lesson_qa_studies: synthetic correctness bench."""
    return _finite_blob(lesson_qa_studies.bench_lesson_qa_studies(seed))
