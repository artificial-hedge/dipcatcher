"""Wave-1599 bench adapters: primate-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    aye_aye_qa_studies,
    howler_qa_studies,
    mouse_lemur_qa_studies,
    night_monkey_qa_studies,
    ring_tailed_qa_studies,
    spider_monkey_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15990


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


def bench_aye_aye_qa_studies_family(seed: int = _SEED + 0):
    """aye_aye_qa_studies: synthetic correctness bench."""
    return _finite_blob(aye_aye_qa_studies.bench_aye_aye_qa_studies(seed))


def bench_howler_qa_studies_family(seed: int = _SEED + 1):
    """howler_qa_studies: synthetic correctness bench."""
    return _finite_blob(howler_qa_studies.bench_howler_qa_studies(seed))


def bench_mouse_lemur_qa_studies_family(seed: int = _SEED + 2):
    """mouse_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(mouse_lemur_qa_studies.bench_mouse_lemur_qa_studies(seed))


def bench_night_monkey_qa_studies_family(seed: int = _SEED + 3):
    """night_monkey_qa_studies: synthetic correctness bench."""
    return _finite_blob(night_monkey_qa_studies.bench_night_monkey_qa_studies(seed))


def bench_ring_tailed_qa_studies_family(seed: int = _SEED + 4):
    """ring_tailed_qa_studies: synthetic correctness bench."""
    return _finite_blob(ring_tailed_qa_studies.bench_ring_tailed_qa_studies(seed))


def bench_spider_monkey_qa_studies_family(seed: int = _SEED + 5):
    """spider_monkey_qa_studies: synthetic correctness bench."""
    return _finite_blob(spider_monkey_qa_studies.bench_spider_monkey_qa_studies(seed))
