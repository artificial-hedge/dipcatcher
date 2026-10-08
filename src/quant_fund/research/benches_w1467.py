"""Wave-1467 bench adapters: highland canon (SYNTHETIC only)."""

from quant_fund.models import (
    arch_qa_studies,
    steppe_qa_studies,
    summit_qa_studies,
    tundra_qa_studies,
    valley_qa_studies,
    volcano_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14670


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


def bench_arch_qa_studies_family(seed: int = _SEED + 0):
    """arch_qa_studies: synthetic correctness bench."""
    return _finite_blob(arch_qa_studies.bench_arch_qa_studies(seed))


def bench_steppe_qa_studies_family(seed: int = _SEED + 1):
    """steppe_qa_studies: synthetic correctness bench."""
    return _finite_blob(steppe_qa_studies.bench_steppe_qa_studies(seed))


def bench_summit_qa_studies_family(seed: int = _SEED + 2):
    """summit_qa_studies: synthetic correctness bench."""
    return _finite_blob(summit_qa_studies.bench_summit_qa_studies(seed))


def bench_tundra_qa_studies_family(seed: int = _SEED + 3):
    """tundra_qa_studies: synthetic correctness bench."""
    return _finite_blob(tundra_qa_studies.bench_tundra_qa_studies(seed))


def bench_valley_qa_studies_family(seed: int = _SEED + 4):
    """valley_qa_studies: synthetic correctness bench."""
    return _finite_blob(valley_qa_studies.bench_valley_qa_studies(seed))


def bench_volcano_qa_studies_family(seed: int = _SEED + 5):
    """volcano_qa_studies: synthetic correctness bench."""
    return _finite_blob(volcano_qa_studies.bench_volcano_qa_studies(seed))
