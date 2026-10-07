"""Wave-1532 bench adapters: riverbird canon (SYNTHETIC only)."""

from quant_fund.models import (
    bee_eater_qa_studies,
    jacamar_qa_studies,
    kookaburra_qa_studies,
    motmot_qa_studies,
    roller_qa_studies,
    tody_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15320


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


def bench_bee_eater_qa_studies_family(seed: int = _SEED + 0):
    """bee_eater_qa_studies: synthetic correctness bench."""
    return _finite_blob(bee_eater_qa_studies.bench_bee_eater_qa_studies(seed))


def bench_jacamar_qa_studies_family(seed: int = _SEED + 1):
    """jacamar_qa_studies: synthetic correctness bench."""
    return _finite_blob(jacamar_qa_studies.bench_jacamar_qa_studies(seed))


def bench_kookaburra_qa_studies_family(seed: int = _SEED + 2):
    """kookaburra_qa_studies: synthetic correctness bench."""
    return _finite_blob(kookaburra_qa_studies.bench_kookaburra_qa_studies(seed))


def bench_motmot_qa_studies_family(seed: int = _SEED + 3):
    """motmot_qa_studies: synthetic correctness bench."""
    return _finite_blob(motmot_qa_studies.bench_motmot_qa_studies(seed))


def bench_roller_qa_studies_family(seed: int = _SEED + 4):
    """roller_qa_studies: synthetic correctness bench."""
    return _finite_blob(roller_qa_studies.bench_roller_qa_studies(seed))


def bench_tody_qa_studies_family(seed: int = _SEED + 5):
    """tody_qa_studies: synthetic correctness bench."""
    return _finite_blob(tody_qa_studies.bench_tody_qa_studies(seed))
