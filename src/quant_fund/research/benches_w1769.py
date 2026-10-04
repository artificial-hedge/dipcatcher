"""Wave-1769 bench adapters: persian-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    angra_qa_studies,
    arash_qa_studies,
    haoma_qa_studies,
    simurgh_qa_studies,
    spenta_qa_studies,
    zal_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17690


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_angra_qa_studies_family(seed: int = _SEED + 0):
    """angra_qa_studies: synthetic correctness bench."""
    return _finite_blob(angra_qa_studies.bench_angra_qa_studies(seed))


def bench_arash_qa_studies_family(seed: int = _SEED + 1):
    """arash_qa_studies: synthetic correctness bench."""
    return _finite_blob(arash_qa_studies.bench_arash_qa_studies(seed))


def bench_haoma_qa_studies_family(seed: int = _SEED + 2):
    """haoma_qa_studies: synthetic correctness bench."""
    return _finite_blob(haoma_qa_studies.bench_haoma_qa_studies(seed))


def bench_simurgh_qa_studies_family(seed: int = _SEED + 3):
    """simurgh_qa_studies: synthetic correctness bench."""
    return _finite_blob(simurgh_qa_studies.bench_simurgh_qa_studies(seed))


def bench_spenta_qa_studies_family(seed: int = _SEED + 4):
    """spenta_qa_studies: synthetic correctness bench."""
    return _finite_blob(spenta_qa_studies.bench_spenta_qa_studies(seed))


def bench_zal_qa_studies_family(seed: int = _SEED + 5):
    """zal_qa_studies: synthetic correctness bench."""
    return _finite_blob(zal_qa_studies.bench_zal_qa_studies(seed))
