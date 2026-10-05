"""Wave-1951 bench adapters: goetic-circle canon (SYNTHETIC only)."""

from quant_fund.models import (
    berith_qa_studies,
    bune_qa_studies,
    forneus_qa_studies,
    glasya_labolas_qa_studies,
    naberius_qa_studies,
    ronove_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19510


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_berith_qa_studies_family(seed: int = _SEED + 0):
    """berith_qa_studies: synthetic correctness bench."""
    return _finite_blob(berith_qa_studies.bench_berith_qa_studies(seed))


def bench_bune_qa_studies_family(seed: int = _SEED + 1):
    """bune_qa_studies: synthetic correctness bench."""
    return _finite_blob(bune_qa_studies.bench_bune_qa_studies(seed))


def bench_forneus_qa_studies_family(seed: int = _SEED + 2):
    """forneus_qa_studies: synthetic correctness bench."""
    return _finite_blob(forneus_qa_studies.bench_forneus_qa_studies(seed))


def bench_glasya_labolas_qa_studies_family(seed: int = _SEED + 3):
    """glasya_labolas_qa_studies: synthetic correctness bench."""
    return _finite_blob(glasya_labolas_qa_studies.bench_glasya_labolas_qa_studies(seed))


def bench_naberius_qa_studies_family(seed: int = _SEED + 4):
    """naberius_qa_studies: synthetic correctness bench."""
    return _finite_blob(naberius_qa_studies.bench_naberius_qa_studies(seed))


def bench_ronove_qa_studies_family(seed: int = _SEED + 5):
    """ronove_qa_studies: synthetic correctness bench."""
    return _finite_blob(ronove_qa_studies.bench_ronove_qa_studies(seed))
