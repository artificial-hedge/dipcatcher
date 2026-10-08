"""Wave-1519 bench adapters: mantis canon (SYNTHETIC only)."""

from quant_fund.models import (
    empusa_qa_studies,
    ghost_mantis_qa_studies,
    mantidfly_qa_studies,
    orchid_mantis_qa_studies,
    praying_mantis_qa_studies,
    shield_mantis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15190


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


def bench_empusa_qa_studies_family(seed: int = _SEED + 0):
    """empusa_qa_studies: synthetic correctness bench."""
    return _finite_blob(empusa_qa_studies.bench_empusa_qa_studies(seed))


def bench_ghost_mantis_qa_studies_family(seed: int = _SEED + 1):
    """ghost_mantis_qa_studies: synthetic correctness bench."""
    return _finite_blob(ghost_mantis_qa_studies.bench_ghost_mantis_qa_studies(seed))


def bench_mantidfly_qa_studies_family(seed: int = _SEED + 2):
    """mantidfly_qa_studies: synthetic correctness bench."""
    return _finite_blob(mantidfly_qa_studies.bench_mantidfly_qa_studies(seed))


def bench_orchid_mantis_qa_studies_family(seed: int = _SEED + 3):
    """orchid_mantis_qa_studies: synthetic correctness bench."""
    return _finite_blob(orchid_mantis_qa_studies.bench_orchid_mantis_qa_studies(seed))


def bench_praying_mantis_qa_studies_family(seed: int = _SEED + 4):
    """praying_mantis_qa_studies: synthetic correctness bench."""
    return _finite_blob(praying_mantis_qa_studies.bench_praying_mantis_qa_studies(seed))


def bench_shield_mantis_qa_studies_family(seed: int = _SEED + 5):
    """shield_mantis_qa_studies: synthetic correctness bench."""
    return _finite_blob(shield_mantis_qa_studies.bench_shield_mantis_qa_studies(seed))
