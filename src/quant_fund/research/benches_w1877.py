"""Wave-1877 bench adapters: garamantian canon (SYNTHETIC only)."""

from quant_fund.models import (
    amayya_qa_studies,
    atete_qa_studies,
    guzil_qa_studies,
    igal_qa_studies,
    tiniri_qa_studies,
    warpon_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18770


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amayya_qa_studies_family(seed: int = _SEED + 0):
    """amayya_qa_studies: synthetic correctness bench."""
    return _finite_blob(amayya_qa_studies.bench_amayya_qa_studies(seed))


def bench_atete_qa_studies_family(seed: int = _SEED + 1):
    """atete_qa_studies: synthetic correctness bench."""
    return _finite_blob(atete_qa_studies.bench_atete_qa_studies(seed))


def bench_guzil_qa_studies_family(seed: int = _SEED + 2):
    """guzil_qa_studies: synthetic correctness bench."""
    return _finite_blob(guzil_qa_studies.bench_guzil_qa_studies(seed))


def bench_igal_qa_studies_family(seed: int = _SEED + 3):
    """igal_qa_studies: synthetic correctness bench."""
    return _finite_blob(igal_qa_studies.bench_igal_qa_studies(seed))


def bench_tiniri_qa_studies_family(seed: int = _SEED + 4):
    """tiniri_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiniri_qa_studies.bench_tiniri_qa_studies(seed))


def bench_warpon_qa_studies_family(seed: int = _SEED + 5):
    """warpon_qa_studies: synthetic correctness bench."""
    return _finite_blob(warpon_qa_studies.bench_warpon_qa_studies(seed))
