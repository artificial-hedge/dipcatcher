"""Wave-1577 bench adapters: carnivore canon (SYNTHETIC only)."""

from quant_fund.models import (
    civet_qa_studies,
    genet_qa_studies,
    manul_qa_studies,
    mongoose_qa_studies,
    sloth_bear_qa_studies,
    suricate_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15770


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


def bench_civet_qa_studies_family(seed: int = _SEED + 0):
    """civet_qa_studies: synthetic correctness bench."""
    return _finite_blob(civet_qa_studies.bench_civet_qa_studies(seed))


def bench_genet_qa_studies_family(seed: int = _SEED + 1):
    """genet_qa_studies: synthetic correctness bench."""
    return _finite_blob(genet_qa_studies.bench_genet_qa_studies(seed))


def bench_manul_qa_studies_family(seed: int = _SEED + 2):
    """manul_qa_studies: synthetic correctness bench."""
    return _finite_blob(manul_qa_studies.bench_manul_qa_studies(seed))


def bench_mongoose_qa_studies_family(seed: int = _SEED + 3):
    """mongoose_qa_studies: synthetic correctness bench."""
    return _finite_blob(mongoose_qa_studies.bench_mongoose_qa_studies(seed))


def bench_sloth_bear_qa_studies_family(seed: int = _SEED + 4):
    """sloth_bear_qa_studies: synthetic correctness bench."""
    return _finite_blob(sloth_bear_qa_studies.bench_sloth_bear_qa_studies(seed))


def bench_suricate_qa_studies_family(seed: int = _SEED + 5):
    """suricate_qa_studies: synthetic correctness bench."""
    return _finite_blob(suricate_qa_studies.bench_suricate_qa_studies(seed))
