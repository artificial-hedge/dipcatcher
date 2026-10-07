"""Wave-1544 bench adapters: coraciiform canon (SYNTHETIC only)."""

from quant_fund.models import (
    hoopoe_qa_studies,
    nunbird_qa_studies,
    nunlet_qa_studies,
    puffbird_qa_studies,
    toco_qa_studies,
    woodhoopoe_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15440


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


def bench_hoopoe_qa_studies_family(seed: int = _SEED + 0):
    """hoopoe_qa_studies: synthetic correctness bench."""
    return _finite_blob(hoopoe_qa_studies.bench_hoopoe_qa_studies(seed))


def bench_nunbird_qa_studies_family(seed: int = _SEED + 1):
    """nunbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(nunbird_qa_studies.bench_nunbird_qa_studies(seed))


def bench_nunlet_qa_studies_family(seed: int = _SEED + 2):
    """nunlet_qa_studies: synthetic correctness bench."""
    return _finite_blob(nunlet_qa_studies.bench_nunlet_qa_studies(seed))


def bench_puffbird_qa_studies_family(seed: int = _SEED + 3):
    """puffbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(puffbird_qa_studies.bench_puffbird_qa_studies(seed))


def bench_toco_qa_studies_family(seed: int = _SEED + 4):
    """toco_qa_studies: synthetic correctness bench."""
    return _finite_blob(toco_qa_studies.bench_toco_qa_studies(seed))


def bench_woodhoopoe_qa_studies_family(seed: int = _SEED + 5):
    """woodhoopoe_qa_studies: synthetic correctness bench."""
    return _finite_blob(woodhoopoe_qa_studies.bench_woodhoopoe_qa_studies(seed))
