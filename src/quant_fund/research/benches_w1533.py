"""Wave-1533 bench adapters: canopybird canon (SYNTHETIC only)."""

from quant_fund.models import (
    aracari_qa_studies,
    barbet_qa_studies,
    honeyguide_qa_studies,
    hornbill_qa_studies,
    quetzal_qa_studies,
    trogon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15330


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aracari_qa_studies_family(seed: int = _SEED + 0):
    """aracari_qa_studies: synthetic correctness bench."""
    return _finite_blob(aracari_qa_studies.bench_aracari_qa_studies(seed))


def bench_barbet_qa_studies_family(seed: int = _SEED + 1):
    """barbet_qa_studies: synthetic correctness bench."""
    return _finite_blob(barbet_qa_studies.bench_barbet_qa_studies(seed))


def bench_honeyguide_qa_studies_family(seed: int = _SEED + 2):
    """honeyguide_qa_studies: synthetic correctness bench."""
    return _finite_blob(honeyguide_qa_studies.bench_honeyguide_qa_studies(seed))


def bench_hornbill_qa_studies_family(seed: int = _SEED + 3):
    """hornbill_qa_studies: synthetic correctness bench."""
    return _finite_blob(hornbill_qa_studies.bench_hornbill_qa_studies(seed))


def bench_quetzal_qa_studies_family(seed: int = _SEED + 4):
    """quetzal_qa_studies: synthetic correctness bench."""
    return _finite_blob(quetzal_qa_studies.bench_quetzal_qa_studies(seed))


def bench_trogon_qa_studies_family(seed: int = _SEED + 5):
    """trogon_qa_studies: synthetic correctness bench."""
    return _finite_blob(trogon_qa_studies.bench_trogon_qa_studies(seed))
