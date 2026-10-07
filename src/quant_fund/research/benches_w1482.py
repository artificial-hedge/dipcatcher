"""Wave-1482 bench adapters: mustelid canon (SYNTHETIC only)."""

from quant_fund.models import (
    ermine_qa_studies,
    fisher_qa_studies,
    marten_qa_studies,
    mink_qa_studies,
    polecat_qa_studies,
    wolverine_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14820


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


def bench_ermine_qa_studies_family(seed: int = _SEED + 0):
    """ermine_qa_studies: synthetic correctness bench."""
    return _finite_blob(ermine_qa_studies.bench_ermine_qa_studies(seed))


def bench_fisher_qa_studies_family(seed: int = _SEED + 1):
    """fisher_qa_studies: synthetic correctness bench."""
    return _finite_blob(fisher_qa_studies.bench_fisher_qa_studies(seed))


def bench_marten_qa_studies_family(seed: int = _SEED + 2):
    """marten_qa_studies: synthetic correctness bench."""
    return _finite_blob(marten_qa_studies.bench_marten_qa_studies(seed))


def bench_mink_qa_studies_family(seed: int = _SEED + 3):
    """mink_qa_studies: synthetic correctness bench."""
    return _finite_blob(mink_qa_studies.bench_mink_qa_studies(seed))


def bench_polecat_qa_studies_family(seed: int = _SEED + 4):
    """polecat_qa_studies: synthetic correctness bench."""
    return _finite_blob(polecat_qa_studies.bench_polecat_qa_studies(seed))


def bench_wolverine_qa_studies_family(seed: int = _SEED + 5):
    """wolverine_qa_studies: synthetic correctness bench."""
    return _finite_blob(wolverine_qa_studies.bench_wolverine_qa_studies(seed))
