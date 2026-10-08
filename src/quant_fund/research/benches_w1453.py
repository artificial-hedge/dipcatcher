"""Wave-1453 bench adapters: raptor canon (SYNTHETIC only)."""

from quant_fund.models import (
    condor_qa_studies,
    harrier_qa_studies,
    kestrel_qa_studies,
    kite_qa_studies,
    osprey_qa_studies,
    vulture_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14530


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


def bench_condor_qa_studies_family(seed: int = _SEED + 0):
    """condor_qa_studies: synthetic correctness bench."""
    return _finite_blob(condor_qa_studies.bench_condor_qa_studies(seed))


def bench_harrier_qa_studies_family(seed: int = _SEED + 1):
    """harrier_qa_studies: synthetic correctness bench."""
    return _finite_blob(harrier_qa_studies.bench_harrier_qa_studies(seed))


def bench_kestrel_qa_studies_family(seed: int = _SEED + 2):
    """kestrel_qa_studies: synthetic correctness bench."""
    return _finite_blob(kestrel_qa_studies.bench_kestrel_qa_studies(seed))


def bench_kite_qa_studies_family(seed: int = _SEED + 3):
    """kite_qa_studies: synthetic correctness bench."""
    return _finite_blob(kite_qa_studies.bench_kite_qa_studies(seed))


def bench_osprey_qa_studies_family(seed: int = _SEED + 4):
    """osprey_qa_studies: synthetic correctness bench."""
    return _finite_blob(osprey_qa_studies.bench_osprey_qa_studies(seed))


def bench_vulture_qa_studies_family(seed: int = _SEED + 5):
    """vulture_qa_studies: synthetic correctness bench."""
    return _finite_blob(vulture_qa_studies.bench_vulture_qa_studies(seed))
