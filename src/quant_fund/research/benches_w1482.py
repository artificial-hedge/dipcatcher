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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
