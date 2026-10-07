"""Wave-1486 bench adapters: wader canon (SYNTHETIC only)."""

from quant_fund.models import (
    flamingo_qa_studies,
    godwit_qa_studies,
    grebe_qa_studies,
    pelican_qa_studies,
    spoonbill_qa_studies,
    stork_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14860


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


def bench_flamingo_qa_studies_family(seed: int = _SEED + 0):
    """flamingo_qa_studies: synthetic correctness bench."""
    return _finite_blob(flamingo_qa_studies.bench_flamingo_qa_studies(seed))


def bench_godwit_qa_studies_family(seed: int = _SEED + 1):
    """godwit_qa_studies: synthetic correctness bench."""
    return _finite_blob(godwit_qa_studies.bench_godwit_qa_studies(seed))


def bench_grebe_qa_studies_family(seed: int = _SEED + 2):
    """grebe_qa_studies: synthetic correctness bench."""
    return _finite_blob(grebe_qa_studies.bench_grebe_qa_studies(seed))


def bench_pelican_qa_studies_family(seed: int = _SEED + 3):
    """pelican_qa_studies: synthetic correctness bench."""
    return _finite_blob(pelican_qa_studies.bench_pelican_qa_studies(seed))


def bench_spoonbill_qa_studies_family(seed: int = _SEED + 4):
    """spoonbill_qa_studies: synthetic correctness bench."""
    return _finite_blob(spoonbill_qa_studies.bench_spoonbill_qa_studies(seed))


def bench_stork_qa_studies_family(seed: int = _SEED + 5):
    """stork_qa_studies: synthetic correctness bench."""
    return _finite_blob(stork_qa_studies.bench_stork_qa_studies(seed))
