"""Wave-1616 bench adapters: plankton-shore canon (SYNTHETIC only)."""

from quant_fund.models import (
    amphipod_qa_studies,
    barnacle_qa_studies,
    copepod_qa_studies,
    isopod_qa_studies,
    krill_qa_studies,
    sandhopper_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16160


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


def bench_amphipod_qa_studies_family(seed: int = _SEED + 0):
    """amphipod_qa_studies: synthetic correctness bench."""
    return _finite_blob(amphipod_qa_studies.bench_amphipod_qa_studies(seed))


def bench_barnacle_qa_studies_family(seed: int = _SEED + 1):
    """barnacle_qa_studies: synthetic correctness bench."""
    return _finite_blob(barnacle_qa_studies.bench_barnacle_qa_studies(seed))


def bench_copepod_qa_studies_family(seed: int = _SEED + 2):
    """copepod_qa_studies: synthetic correctness bench."""
    return _finite_blob(copepod_qa_studies.bench_copepod_qa_studies(seed))


def bench_isopod_qa_studies_family(seed: int = _SEED + 3):
    """isopod_qa_studies: synthetic correctness bench."""
    return _finite_blob(isopod_qa_studies.bench_isopod_qa_studies(seed))


def bench_krill_qa_studies_family(seed: int = _SEED + 4):
    """krill_qa_studies: synthetic correctness bench."""
    return _finite_blob(krill_qa_studies.bench_krill_qa_studies(seed))


def bench_sandhopper_qa_studies_family(seed: int = _SEED + 5):
    """sandhopper_qa_studies: synthetic correctness bench."""
    return _finite_blob(sandhopper_qa_studies.bench_sandhopper_qa_studies(seed))
