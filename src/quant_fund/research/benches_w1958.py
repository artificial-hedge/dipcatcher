"""Wave-1958 bench adapters: goetic-hierarchy canon (SYNTHETIC only)."""

from quant_fund.models import (
    bathin_qa_studies,
    buer_qa_studies,
    marax_qa_studies,
    marbas_qa_studies,
    purson_qa_studies,
    sallos_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19580


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bathin_qa_studies_family(seed: int = _SEED + 0):
    """bathin_qa_studies: synthetic correctness bench."""
    return _finite_blob(bathin_qa_studies.bench_bathin_qa_studies(seed))


def bench_buer_qa_studies_family(seed: int = _SEED + 1):
    """buer_qa_studies: synthetic correctness bench."""
    return _finite_blob(buer_qa_studies.bench_buer_qa_studies(seed))


def bench_marax_qa_studies_family(seed: int = _SEED + 2):
    """marax_qa_studies: synthetic correctness bench."""
    return _finite_blob(marax_qa_studies.bench_marax_qa_studies(seed))


def bench_marbas_qa_studies_family(seed: int = _SEED + 3):
    """marbas_qa_studies: synthetic correctness bench."""
    return _finite_blob(marbas_qa_studies.bench_marbas_qa_studies(seed))


def bench_purson_qa_studies_family(seed: int = _SEED + 4):
    """purson_qa_studies: synthetic correctness bench."""
    return _finite_blob(purson_qa_studies.bench_purson_qa_studies(seed))


def bench_sallos_qa_studies_family(seed: int = _SEED + 5):
    """sallos_qa_studies: synthetic correctness bench."""
    return _finite_blob(sallos_qa_studies.bench_sallos_qa_studies(seed))
