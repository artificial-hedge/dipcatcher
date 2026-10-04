"""Wave-1715 bench adapters: dacian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    bendis_qa_studies,
    darzalas_qa_studies,
    derzelas_qa_studies,
    kezion_qa_studies,
    sabazios_qa_studies,
    zamolxis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17150


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bendis_qa_studies_family(seed: int = _SEED + 0):
    """bendis_qa_studies: synthetic correctness bench."""
    return _finite_blob(bendis_qa_studies.bench_bendis_qa_studies(seed))


def bench_darzalas_qa_studies_family(seed: int = _SEED + 1):
    """darzalas_qa_studies: synthetic correctness bench."""
    return _finite_blob(darzalas_qa_studies.bench_darzalas_qa_studies(seed))


def bench_derzelas_qa_studies_family(seed: int = _SEED + 2):
    """derzelas_qa_studies: synthetic correctness bench."""
    return _finite_blob(derzelas_qa_studies.bench_derzelas_qa_studies(seed))


def bench_kezion_qa_studies_family(seed: int = _SEED + 3):
    """kezion_qa_studies: synthetic correctness bench."""
    return _finite_blob(kezion_qa_studies.bench_kezion_qa_studies(seed))


def bench_sabazios_qa_studies_family(seed: int = _SEED + 4):
    """sabazios_qa_studies: synthetic correctness bench."""
    return _finite_blob(sabazios_qa_studies.bench_sabazios_qa_studies(seed))


def bench_zamolxis_qa_studies_family(seed: int = _SEED + 5):
    """zamolxis_qa_studies: synthetic correctness bench."""
    return _finite_blob(zamolxis_qa_studies.bench_zamolxis_qa_studies(seed))
