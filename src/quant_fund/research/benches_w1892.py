"""Wave-1892 bench adapters: folk-spirit lore-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    al_basti_qa_studies,
    ananke_libya_qa_studies,
    encantado_qa_studies,
    mithra_iran_qa_studies,
    mitra_persian_qa_studies,
    perangal_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18920


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_al_basti_qa_studies_family(seed: int = _SEED + 0):
    """al_basti_qa_studies: synthetic correctness bench."""
    return _finite_blob(al_basti_qa_studies.bench_al_basti_qa_studies(seed))


def bench_ananke_libya_qa_studies_family(seed: int = _SEED + 1):
    """ananke_libya_qa_studies: synthetic correctness bench."""
    return _finite_blob(ananke_libya_qa_studies.bench_ananke_libya_qa_studies(seed))


def bench_encantado_qa_studies_family(seed: int = _SEED + 2):
    """encantado_qa_studies: synthetic correctness bench."""
    return _finite_blob(encantado_qa_studies.bench_encantado_qa_studies(seed))


def bench_mithra_iran_qa_studies_family(seed: int = _SEED + 3):
    """mithra_iran_qa_studies: synthetic correctness bench."""
    return _finite_blob(mithra_iran_qa_studies.bench_mithra_iran_qa_studies(seed))


def bench_mitra_persian_qa_studies_family(seed: int = _SEED + 4):
    """mitra_persian_qa_studies: synthetic correctness bench."""
    return _finite_blob(mitra_persian_qa_studies.bench_mitra_persian_qa_studies(seed))


def bench_perangal_qa_studies_family(seed: int = _SEED + 5):
    """perangal_qa_studies: synthetic correctness bench."""
    return _finite_blob(perangal_qa_studies.bench_perangal_qa_studies(seed))
