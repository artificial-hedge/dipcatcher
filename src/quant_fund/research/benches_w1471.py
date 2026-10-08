"""Wave-1471 bench adapters: wildflower canon (SYNTHETIC only)."""

from quant_fund.models import (
    crocus_qa_studies,
    daffodil_qa_studies,
    daisy_qa_studies,
    foxglove_qa_studies,
    iris_qa_studies,
    poppy_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14710


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


def bench_crocus_qa_studies_family(seed: int = _SEED + 0):
    """crocus_qa_studies: synthetic correctness bench."""
    return _finite_blob(crocus_qa_studies.bench_crocus_qa_studies(seed))


def bench_daffodil_qa_studies_family(seed: int = _SEED + 1):
    """daffodil_qa_studies: synthetic correctness bench."""
    return _finite_blob(daffodil_qa_studies.bench_daffodil_qa_studies(seed))


def bench_daisy_qa_studies_family(seed: int = _SEED + 2):
    """daisy_qa_studies: synthetic correctness bench."""
    return _finite_blob(daisy_qa_studies.bench_daisy_qa_studies(seed))


def bench_foxglove_qa_studies_family(seed: int = _SEED + 3):
    """foxglove_qa_studies: synthetic correctness bench."""
    return _finite_blob(foxglove_qa_studies.bench_foxglove_qa_studies(seed))


def bench_iris_qa_studies_family(seed: int = _SEED + 4):
    """iris_qa_studies: synthetic correctness bench."""
    return _finite_blob(iris_qa_studies.bench_iris_qa_studies(seed))


def bench_poppy_qa_studies_family(seed: int = _SEED + 5):
    """poppy_qa_studies: synthetic correctness bench."""
    return _finite_blob(poppy_qa_studies.bench_poppy_qa_studies(seed))
