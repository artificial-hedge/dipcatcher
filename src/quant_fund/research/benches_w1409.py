"""Wave-1409 bench adapters: multi-hop-QA-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bamboogle_lite_studies,
    beerqa_lite_studies,
    cider_qa_studies,
    ensem_qa_studies,
    fanqa_lite_studies,
    hops_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14090


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


def bench_bamboogle_lite_studies_family(seed: int = _SEED + 0):
    """bamboogle_lite_studies: synthetic correctness bench."""
    return _finite_blob(bamboogle_lite_studies.bench_bamboogle_lite_studies(seed))


def bench_beerqa_lite_studies_family(seed: int = _SEED + 1):
    """beerqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(beerqa_lite_studies.bench_beerqa_lite_studies(seed))


def bench_cider_qa_studies_family(seed: int = _SEED + 2):
    """cider_qa_studies: synthetic correctness bench."""
    return _finite_blob(cider_qa_studies.bench_cider_qa_studies(seed))


def bench_ensem_qa_studies_family(seed: int = _SEED + 3):
    """ensem_qa_studies: synthetic correctness bench."""
    return _finite_blob(ensem_qa_studies.bench_ensem_qa_studies(seed))


def bench_fanqa_lite_studies_family(seed: int = _SEED + 4):
    """fanqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(fanqa_lite_studies.bench_fanqa_lite_studies(seed))


def bench_hops_qa_studies_family(seed: int = _SEED + 5):
    """hops_qa_studies: synthetic correctness bench."""
    return _finite_blob(hops_qa_studies.bench_hops_qa_studies(seed))
