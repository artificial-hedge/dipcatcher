"""Wave-1491 bench adapters: tree canon (SYNTHETIC only)."""

from quant_fund.models import (
    cypress_qa_studies,
    eucalyptus_qa_studies,
    hemlock_qa_studies,
    laurel_qa_studies,
    magnolia_qa_studies,
    spruce_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14910


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cypress_qa_studies_family(seed: int = _SEED + 0):
    """cypress_qa_studies: synthetic correctness bench."""
    return _finite_blob(cypress_qa_studies.bench_cypress_qa_studies(seed))


def bench_eucalyptus_qa_studies_family(seed: int = _SEED + 1):
    """eucalyptus_qa_studies: synthetic correctness bench."""
    return _finite_blob(eucalyptus_qa_studies.bench_eucalyptus_qa_studies(seed))


def bench_hemlock_qa_studies_family(seed: int = _SEED + 2):
    """hemlock_qa_studies: synthetic correctness bench."""
    return _finite_blob(hemlock_qa_studies.bench_hemlock_qa_studies(seed))


def bench_laurel_qa_studies_family(seed: int = _SEED + 3):
    """laurel_qa_studies: synthetic correctness bench."""
    return _finite_blob(laurel_qa_studies.bench_laurel_qa_studies(seed))


def bench_magnolia_qa_studies_family(seed: int = _SEED + 4):
    """magnolia_qa_studies: synthetic correctness bench."""
    return _finite_blob(magnolia_qa_studies.bench_magnolia_qa_studies(seed))


def bench_spruce_qa_studies_family(seed: int = _SEED + 5):
    """spruce_qa_studies: synthetic correctness bench."""
    return _finite_blob(spruce_qa_studies.bench_spruce_qa_studies(seed))
