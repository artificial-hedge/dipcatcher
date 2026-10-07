"""Wave-1346 bench adapters: KB-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    grail_qa_studies,
    graph_questions_studies,
    kqa_pro_studies,
    lc_quad_studies,
    mintaka_qa_studies,
    spinach_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13460


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


def bench_grail_qa_studies_family(seed: int = _SEED + 0):
    """grail_qa_studies: synthetic correctness bench."""
    return _finite_blob(grail_qa_studies.bench_grail_qa_studies(seed))


def bench_graph_questions_studies_family(seed: int = _SEED + 1):
    """graph_questions_studies: synthetic correctness bench."""
    return _finite_blob(graph_questions_studies.bench_graph_questions_studies(seed))


def bench_kqa_pro_studies_family(seed: int = _SEED + 2):
    """kqa_pro_studies: synthetic correctness bench."""
    return _finite_blob(kqa_pro_studies.bench_kqa_pro_studies(seed))


def bench_lc_quad_studies_family(seed: int = _SEED + 3):
    """lc_quad_studies: synthetic correctness bench."""
    return _finite_blob(lc_quad_studies.bench_lc_quad_studies(seed))


def bench_mintaka_qa_studies_family(seed: int = _SEED + 4):
    """mintaka_qa_studies: synthetic correctness bench."""
    return _finite_blob(mintaka_qa_studies.bench_mintaka_qa_studies(seed))


def bench_spinach_qa_studies_family(seed: int = _SEED + 5):
    """spinach_qa_studies: synthetic correctness bench."""
    return _finite_blob(spinach_qa_studies.bench_spinach_qa_studies(seed))
