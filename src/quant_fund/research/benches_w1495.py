"""Wave-1495 bench adapters: mustelid-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    grison_qa_studies,
    sable_qa_studies,
    stoat_qa_studies,
    tayra_qa_studies,
    weasel_qa_studies,
    zorilla_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14950


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


def bench_grison_qa_studies_family(seed: int = _SEED + 0):
    """grison_qa_studies: synthetic correctness bench."""
    return _finite_blob(grison_qa_studies.bench_grison_qa_studies(seed))


def bench_sable_qa_studies_family(seed: int = _SEED + 1):
    """sable_qa_studies: synthetic correctness bench."""
    return _finite_blob(sable_qa_studies.bench_sable_qa_studies(seed))


def bench_stoat_qa_studies_family(seed: int = _SEED + 2):
    """stoat_qa_studies: synthetic correctness bench."""
    return _finite_blob(stoat_qa_studies.bench_stoat_qa_studies(seed))


def bench_tayra_qa_studies_family(seed: int = _SEED + 3):
    """tayra_qa_studies: synthetic correctness bench."""
    return _finite_blob(tayra_qa_studies.bench_tayra_qa_studies(seed))


def bench_weasel_qa_studies_family(seed: int = _SEED + 4):
    """weasel_qa_studies: synthetic correctness bench."""
    return _finite_blob(weasel_qa_studies.bench_weasel_qa_studies(seed))


def bench_zorilla_qa_studies_family(seed: int = _SEED + 5):
    """zorilla_qa_studies: synthetic correctness bench."""
    return _finite_blob(zorilla_qa_studies.bench_zorilla_qa_studies(seed))
