"""Wave-1894 bench adapters: mesopotamian-demon-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    asakku_qa_studies,
    ekimmu_qa_studies,
    etemmu_qa_studies,
    gidim_qa_studies,
    maskim_qa_studies,
    sebettu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18940


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_asakku_qa_studies_family(seed: int = _SEED + 0):
    """asakku_qa_studies: synthetic correctness bench."""
    return _finite_blob(asakku_qa_studies.bench_asakku_qa_studies(seed))


def bench_ekimmu_qa_studies_family(seed: int = _SEED + 1):
    """ekimmu_qa_studies: synthetic correctness bench."""
    return _finite_blob(ekimmu_qa_studies.bench_ekimmu_qa_studies(seed))


def bench_etemmu_qa_studies_family(seed: int = _SEED + 2):
    """etemmu_qa_studies: synthetic correctness bench."""
    return _finite_blob(etemmu_qa_studies.bench_etemmu_qa_studies(seed))


def bench_gidim_qa_studies_family(seed: int = _SEED + 3):
    """gidim_qa_studies: synthetic correctness bench."""
    return _finite_blob(gidim_qa_studies.bench_gidim_qa_studies(seed))


def bench_maskim_qa_studies_family(seed: int = _SEED + 4):
    """maskim_qa_studies: synthetic correctness bench."""
    return _finite_blob(maskim_qa_studies.bench_maskim_qa_studies(seed))


def bench_sebettu_qa_studies_family(seed: int = _SEED + 5):
    """sebettu_qa_studies: synthetic correctness bench."""
    return _finite_blob(sebettu_qa_studies.bench_sebettu_qa_studies(seed))
