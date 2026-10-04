"""Wave-1609 bench adapters: lemur-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    black_lemur_qa_studies,
    brown_lemur_qa_studies,
    dwarf_lemur_qa_studies,
    mongoose_lemur_qa_studies,
    ruffed_qa_studies,
    sportive_lemur_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16090


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_black_lemur_qa_studies_family(seed: int = _SEED + 0):
    """black_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(black_lemur_qa_studies.bench_black_lemur_qa_studies(seed))


def bench_brown_lemur_qa_studies_family(seed: int = _SEED + 1):
    """brown_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(brown_lemur_qa_studies.bench_brown_lemur_qa_studies(seed))


def bench_dwarf_lemur_qa_studies_family(seed: int = _SEED + 2):
    """dwarf_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(dwarf_lemur_qa_studies.bench_dwarf_lemur_qa_studies(seed))


def bench_mongoose_lemur_qa_studies_family(seed: int = _SEED + 3):
    """mongoose_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(mongoose_lemur_qa_studies.bench_mongoose_lemur_qa_studies(seed))


def bench_ruffed_qa_studies_family(seed: int = _SEED + 4):
    """ruffed_qa_studies: synthetic correctness bench."""
    return _finite_blob(ruffed_qa_studies.bench_ruffed_qa_studies(seed))


def bench_sportive_lemur_qa_studies_family(seed: int = _SEED + 5):
    """sportive_lemur_qa_studies: synthetic correctness bench."""
    return _finite_blob(sportive_lemur_qa_studies.bench_sportive_lemur_qa_studies(seed))
