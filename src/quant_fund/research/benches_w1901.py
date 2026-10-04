"""Wave-1901 bench adapters: brazilian-folklore canon (SYNTHETIC only)."""

from quant_fund.models import (
    boitata_qa_studies,
    boto_qa_studies,
    curupira_qa_studies,
    iara_qa_studies,
    mapinguari_qa_studies,
    saci_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19010


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_boitata_qa_studies_family(seed: int = _SEED + 0):
    """boitata_qa_studies: synthetic correctness bench."""
    return _finite_blob(boitata_qa_studies.bench_boitata_qa_studies(seed))


def bench_boto_qa_studies_family(seed: int = _SEED + 1):
    """boto_qa_studies: synthetic correctness bench."""
    return _finite_blob(boto_qa_studies.bench_boto_qa_studies(seed))


def bench_curupira_qa_studies_family(seed: int = _SEED + 2):
    """curupira_qa_studies: synthetic correctness bench."""
    return _finite_blob(curupira_qa_studies.bench_curupira_qa_studies(seed))


def bench_iara_qa_studies_family(seed: int = _SEED + 3):
    """iara_qa_studies: synthetic correctness bench."""
    return _finite_blob(iara_qa_studies.bench_iara_qa_studies(seed))


def bench_mapinguari_qa_studies_family(seed: int = _SEED + 4):
    """mapinguari_qa_studies: synthetic correctness bench."""
    return _finite_blob(mapinguari_qa_studies.bench_mapinguari_qa_studies(seed))


def bench_saci_qa_studies_family(seed: int = _SEED + 5):
    """saci_qa_studies: synthetic correctness bench."""
    return _finite_blob(saci_qa_studies.bench_saci_qa_studies(seed))
