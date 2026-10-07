"""Wave-1571 bench adapters: bivalve canon (SYNTHETIC only)."""

from quant_fund.models import (
    clam_qa_studies,
    conch_qa_studies,
    mussel_qa_studies,
    oyster_qa_studies,
    scallop_qa_studies,
    whelk_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15710


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


def bench_clam_qa_studies_family(seed: int = _SEED + 0):
    """clam_qa_studies: synthetic correctness bench."""
    return _finite_blob(clam_qa_studies.bench_clam_qa_studies(seed))


def bench_conch_qa_studies_family(seed: int = _SEED + 1):
    """conch_qa_studies: synthetic correctness bench."""
    return _finite_blob(conch_qa_studies.bench_conch_qa_studies(seed))


def bench_mussel_qa_studies_family(seed: int = _SEED + 2):
    """mussel_qa_studies: synthetic correctness bench."""
    return _finite_blob(mussel_qa_studies.bench_mussel_qa_studies(seed))


def bench_oyster_qa_studies_family(seed: int = _SEED + 3):
    """oyster_qa_studies: synthetic correctness bench."""
    return _finite_blob(oyster_qa_studies.bench_oyster_qa_studies(seed))


def bench_scallop_qa_studies_family(seed: int = _SEED + 4):
    """scallop_qa_studies: synthetic correctness bench."""
    return _finite_blob(scallop_qa_studies.bench_scallop_qa_studies(seed))


def bench_whelk_qa_studies_family(seed: int = _SEED + 5):
    """whelk_qa_studies: synthetic correctness bench."""
    return _finite_blob(whelk_qa_studies.bench_whelk_qa_studies(seed))
