"""Wave-1443 bench adapters: gem canon (SYNTHETIC only)."""

from quant_fund.models import (
    amber_qa_studies,
    amethyst_qa_studies,
    crystal_qa_studies,
    diamond_qa_studies,
    emerald_qa_studies,
    jade_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14430


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


def bench_amber_qa_studies_family(seed: int = _SEED + 0):
    """amber_qa_studies: synthetic correctness bench."""
    return _finite_blob(amber_qa_studies.bench_amber_qa_studies(seed))


def bench_amethyst_qa_studies_family(seed: int = _SEED + 1):
    """amethyst_qa_studies: synthetic correctness bench."""
    return _finite_blob(amethyst_qa_studies.bench_amethyst_qa_studies(seed))


def bench_crystal_qa_studies_family(seed: int = _SEED + 2):
    """crystal_qa_studies: synthetic correctness bench."""
    return _finite_blob(crystal_qa_studies.bench_crystal_qa_studies(seed))


def bench_diamond_qa_studies_family(seed: int = _SEED + 3):
    """diamond_qa_studies: synthetic correctness bench."""
    return _finite_blob(diamond_qa_studies.bench_diamond_qa_studies(seed))


def bench_emerald_qa_studies_family(seed: int = _SEED + 4):
    """emerald_qa_studies: synthetic correctness bench."""
    return _finite_blob(emerald_qa_studies.bench_emerald_qa_studies(seed))


def bench_jade_qa_studies_family(seed: int = _SEED + 5):
    """jade_qa_studies: synthetic correctness bench."""
    return _finite_blob(jade_qa_studies.bench_jade_qa_studies(seed))
