"""Wave-1563 bench adapters: crustacean canon (SYNTHETIC only)."""

from quant_fund.models import (
    cleaner_shrimp_qa_studies,
    decorator_crab_qa_studies,
    hermit_crab_qa_studies,
    mantis_shrimp_qa_studies,
    pistol_shrimp_qa_studies,
    porcelain_crab_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15630


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


def bench_cleaner_shrimp_qa_studies_family(seed: int = _SEED + 0):
    """cleaner_shrimp_qa_studies: synthetic correctness bench."""
    return _finite_blob(cleaner_shrimp_qa_studies.bench_cleaner_shrimp_qa_studies(seed))


def bench_decorator_crab_qa_studies_family(seed: int = _SEED + 1):
    """decorator_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(decorator_crab_qa_studies.bench_decorator_crab_qa_studies(seed))


def bench_hermit_crab_qa_studies_family(seed: int = _SEED + 2):
    """hermit_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(hermit_crab_qa_studies.bench_hermit_crab_qa_studies(seed))


def bench_mantis_shrimp_qa_studies_family(seed: int = _SEED + 3):
    """mantis_shrimp_qa_studies: synthetic correctness bench."""
    return _finite_blob(mantis_shrimp_qa_studies.bench_mantis_shrimp_qa_studies(seed))


def bench_pistol_shrimp_qa_studies_family(seed: int = _SEED + 4):
    """pistol_shrimp_qa_studies: synthetic correctness bench."""
    return _finite_blob(pistol_shrimp_qa_studies.bench_pistol_shrimp_qa_studies(seed))


def bench_porcelain_crab_qa_studies_family(seed: int = _SEED + 5):
    """porcelain_crab_qa_studies: synthetic correctness bench."""
    return _finite_blob(porcelain_crab_qa_studies.bench_porcelain_crab_qa_studies(seed))
