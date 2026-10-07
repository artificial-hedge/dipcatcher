"""Wave-1526 bench adapters: fern canon (SYNTHETIC only)."""

from quant_fund.models import (
    bracken_qa_studies,
    horsetail_qa_studies,
    maidenhair_qa_studies,
    staghorn_qa_studies,
    swordfern_qa_studies,
    treefern_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15260


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


def bench_bracken_qa_studies_family(seed: int = _SEED + 0):
    """bracken_qa_studies: synthetic correctness bench."""
    return _finite_blob(bracken_qa_studies.bench_bracken_qa_studies(seed))


def bench_horsetail_qa_studies_family(seed: int = _SEED + 1):
    """horsetail_qa_studies: synthetic correctness bench."""
    return _finite_blob(horsetail_qa_studies.bench_horsetail_qa_studies(seed))


def bench_maidenhair_qa_studies_family(seed: int = _SEED + 2):
    """maidenhair_qa_studies: synthetic correctness bench."""
    return _finite_blob(maidenhair_qa_studies.bench_maidenhair_qa_studies(seed))


def bench_staghorn_qa_studies_family(seed: int = _SEED + 3):
    """staghorn_qa_studies: synthetic correctness bench."""
    return _finite_blob(staghorn_qa_studies.bench_staghorn_qa_studies(seed))


def bench_swordfern_qa_studies_family(seed: int = _SEED + 4):
    """swordfern_qa_studies: synthetic correctness bench."""
    return _finite_blob(swordfern_qa_studies.bench_swordfern_qa_studies(seed))


def bench_treefern_qa_studies_family(seed: int = _SEED + 5):
    """treefern_qa_studies: synthetic correctness bench."""
    return _finite_blob(treefern_qa_studies.bench_treefern_qa_studies(seed))
