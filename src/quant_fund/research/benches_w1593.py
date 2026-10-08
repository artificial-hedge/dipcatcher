"""Wave-1593 bench adapters: old-world-monkey canon (SYNTHETIC only)."""

from quant_fund.models import (
    colobus_qa_studies,
    drill_qa_studies,
    gelada_qa_studies,
    guenon_qa_studies,
    mandrill_qa_studies,
    mangabey_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15930


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


def bench_colobus_qa_studies_family(seed: int = _SEED + 0):
    """colobus_qa_studies: synthetic correctness bench."""
    return _finite_blob(colobus_qa_studies.bench_colobus_qa_studies(seed))


def bench_drill_qa_studies_family(seed: int = _SEED + 1):
    """drill_qa_studies: synthetic correctness bench."""
    return _finite_blob(drill_qa_studies.bench_drill_qa_studies(seed))


def bench_gelada_qa_studies_family(seed: int = _SEED + 2):
    """gelada_qa_studies: synthetic correctness bench."""
    return _finite_blob(gelada_qa_studies.bench_gelada_qa_studies(seed))


def bench_guenon_qa_studies_family(seed: int = _SEED + 3):
    """guenon_qa_studies: synthetic correctness bench."""
    return _finite_blob(guenon_qa_studies.bench_guenon_qa_studies(seed))


def bench_mandrill_qa_studies_family(seed: int = _SEED + 4):
    """mandrill_qa_studies: synthetic correctness bench."""
    return _finite_blob(mandrill_qa_studies.bench_mandrill_qa_studies(seed))


def bench_mangabey_qa_studies_family(seed: int = _SEED + 5):
    """mangabey_qa_studies: synthetic correctness bench."""
    return _finite_blob(mangabey_qa_studies.bench_mangabey_qa_studies(seed))
