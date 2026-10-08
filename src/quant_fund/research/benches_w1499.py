"""Wave-1499 bench adapters: spice-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    oregano_qa_studies,
    parsley_qa_studies,
    rosemary_qa_studies,
    saffron_qa_studies,
    tarragon_qa_studies,
    turmeric_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14990


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


def bench_oregano_qa_studies_family(seed: int = _SEED + 0):
    """oregano_qa_studies: synthetic correctness bench."""
    return _finite_blob(oregano_qa_studies.bench_oregano_qa_studies(seed))


def bench_parsley_qa_studies_family(seed: int = _SEED + 1):
    """parsley_qa_studies: synthetic correctness bench."""
    return _finite_blob(parsley_qa_studies.bench_parsley_qa_studies(seed))


def bench_rosemary_qa_studies_family(seed: int = _SEED + 2):
    """rosemary_qa_studies: synthetic correctness bench."""
    return _finite_blob(rosemary_qa_studies.bench_rosemary_qa_studies(seed))


def bench_saffron_qa_studies_family(seed: int = _SEED + 3):
    """saffron_qa_studies: synthetic correctness bench."""
    return _finite_blob(saffron_qa_studies.bench_saffron_qa_studies(seed))


def bench_tarragon_qa_studies_family(seed: int = _SEED + 4):
    """tarragon_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarragon_qa_studies.bench_tarragon_qa_studies(seed))


def bench_turmeric_qa_studies_family(seed: int = _SEED + 5):
    """turmeric_qa_studies: synthetic correctness bench."""
    return _finite_blob(turmeric_qa_studies.bench_turmeric_qa_studies(seed))
