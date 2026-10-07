"""Wave-1447 bench adapters: predator canon (SYNTHETIC only)."""

from quant_fund.models import (
    bear_qa_studies,
    cheetah_qa_studies,
    fox_qa_studies,
    leopard_qa_studies,
    lion_qa_studies,
    wolf_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14470


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


def bench_bear_qa_studies_family(seed: int = _SEED + 0):
    """bear_qa_studies: synthetic correctness bench."""
    return _finite_blob(bear_qa_studies.bench_bear_qa_studies(seed))


def bench_cheetah_qa_studies_family(seed: int = _SEED + 1):
    """cheetah_qa_studies: synthetic correctness bench."""
    return _finite_blob(cheetah_qa_studies.bench_cheetah_qa_studies(seed))


def bench_fox_qa_studies_family(seed: int = _SEED + 2):
    """fox_qa_studies: synthetic correctness bench."""
    return _finite_blob(fox_qa_studies.bench_fox_qa_studies(seed))


def bench_leopard_qa_studies_family(seed: int = _SEED + 3):
    """leopard_qa_studies: synthetic correctness bench."""
    return _finite_blob(leopard_qa_studies.bench_leopard_qa_studies(seed))


def bench_lion_qa_studies_family(seed: int = _SEED + 4):
    """lion_qa_studies: synthetic correctness bench."""
    return _finite_blob(lion_qa_studies.bench_lion_qa_studies(seed))


def bench_wolf_qa_studies_family(seed: int = _SEED + 5):
    """wolf_qa_studies: synthetic correctness bench."""
    return _finite_blob(wolf_qa_studies.bench_wolf_qa_studies(seed))
