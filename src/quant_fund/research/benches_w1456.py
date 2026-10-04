"""Wave-1456 bench adapters: fish canon (SYNTHETIC only)."""

from quant_fund.models import (
    barracuda_qa_studies,
    catfish_qa_studies,
    cod_qa_studies,
    piranha_qa_studies,
    salmon_qa_studies,
    tuna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14560


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_barracuda_qa_studies_family(seed: int = _SEED + 0):
    """barracuda_qa_studies: synthetic correctness bench."""
    return _finite_blob(barracuda_qa_studies.bench_barracuda_qa_studies(seed))


def bench_catfish_qa_studies_family(seed: int = _SEED + 1):
    """catfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(catfish_qa_studies.bench_catfish_qa_studies(seed))


def bench_cod_qa_studies_family(seed: int = _SEED + 2):
    """cod_qa_studies: synthetic correctness bench."""
    return _finite_blob(cod_qa_studies.bench_cod_qa_studies(seed))


def bench_piranha_qa_studies_family(seed: int = _SEED + 3):
    """piranha_qa_studies: synthetic correctness bench."""
    return _finite_blob(piranha_qa_studies.bench_piranha_qa_studies(seed))


def bench_salmon_qa_studies_family(seed: int = _SEED + 4):
    """salmon_qa_studies: synthetic correctness bench."""
    return _finite_blob(salmon_qa_studies.bench_salmon_qa_studies(seed))


def bench_tuna_qa_studies_family(seed: int = _SEED + 5):
    """tuna_qa_studies: synthetic correctness bench."""
    return _finite_blob(tuna_qa_studies.bench_tuna_qa_studies(seed))
