"""Wave-1903 bench adapters: slavic-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    chert_qa_studies,
    likho_qa_studies,
    polevoy_qa_studies,
    rarog_qa_studies,
    vodyanoy_qa_studies,
    zmey_gorynych_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19030


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_chert_qa_studies_family(seed: int = _SEED + 0):
    """chert_qa_studies: synthetic correctness bench."""
    return _finite_blob(chert_qa_studies.bench_chert_qa_studies(seed))


def bench_likho_qa_studies_family(seed: int = _SEED + 1):
    """likho_qa_studies: synthetic correctness bench."""
    return _finite_blob(likho_qa_studies.bench_likho_qa_studies(seed))


def bench_polevoy_qa_studies_family(seed: int = _SEED + 2):
    """polevoy_qa_studies: synthetic correctness bench."""
    return _finite_blob(polevoy_qa_studies.bench_polevoy_qa_studies(seed))


def bench_rarog_qa_studies_family(seed: int = _SEED + 3):
    """rarog_qa_studies: synthetic correctness bench."""
    return _finite_blob(rarog_qa_studies.bench_rarog_qa_studies(seed))


def bench_vodyanoy_qa_studies_family(seed: int = _SEED + 4):
    """vodyanoy_qa_studies: synthetic correctness bench."""
    return _finite_blob(vodyanoy_qa_studies.bench_vodyanoy_qa_studies(seed))


def bench_zmey_gorynych_qa_studies_family(seed: int = _SEED + 5):
    """zmey_gorynych_qa_studies: synthetic correctness bench."""
    return _finite_blob(zmey_gorynych_qa_studies.bench_zmey_gorynych_qa_studies(seed))
