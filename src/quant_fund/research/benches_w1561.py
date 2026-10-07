"""Wave-1561 bench adapters: reef-fish-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    angelfish_qa_studies,
    blenny_qa_studies,
    goby_qa_studies,
    lionfish_qa_studies,
    surgeonfish_qa_studies,
    triggerfish_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15610


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


def bench_angelfish_qa_studies_family(seed: int = _SEED + 0):
    """angelfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(angelfish_qa_studies.bench_angelfish_qa_studies(seed))


def bench_blenny_qa_studies_family(seed: int = _SEED + 1):
    """blenny_qa_studies: synthetic correctness bench."""
    return _finite_blob(blenny_qa_studies.bench_blenny_qa_studies(seed))


def bench_goby_qa_studies_family(seed: int = _SEED + 2):
    """goby_qa_studies: synthetic correctness bench."""
    return _finite_blob(goby_qa_studies.bench_goby_qa_studies(seed))


def bench_lionfish_qa_studies_family(seed: int = _SEED + 3):
    """lionfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(lionfish_qa_studies.bench_lionfish_qa_studies(seed))


def bench_surgeonfish_qa_studies_family(seed: int = _SEED + 4):
    """surgeonfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(surgeonfish_qa_studies.bench_surgeonfish_qa_studies(seed))


def bench_triggerfish_qa_studies_family(seed: int = _SEED + 5):
    """triggerfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(triggerfish_qa_studies.bench_triggerfish_qa_studies(seed))
