"""Wave-1723 bench adapters: norse-myth-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    honir_qa_studies,
    kvasir_qa_studies,
    lodurr_qa_studies,
    mimir_qa_studies,
    ve_qa_studies,
    vili_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17230


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_honir_qa_studies_family(seed: int = _SEED + 0):
    """honir_qa_studies: synthetic correctness bench."""
    return _finite_blob(honir_qa_studies.bench_honir_qa_studies(seed))


def bench_kvasir_qa_studies_family(seed: int = _SEED + 1):
    """kvasir_qa_studies: synthetic correctness bench."""
    return _finite_blob(kvasir_qa_studies.bench_kvasir_qa_studies(seed))


def bench_lodurr_qa_studies_family(seed: int = _SEED + 2):
    """lodurr_qa_studies: synthetic correctness bench."""
    return _finite_blob(lodurr_qa_studies.bench_lodurr_qa_studies(seed))


def bench_mimir_qa_studies_family(seed: int = _SEED + 3):
    """mimir_qa_studies: synthetic correctness bench."""
    return _finite_blob(mimir_qa_studies.bench_mimir_qa_studies(seed))


def bench_ve_qa_studies_family(seed: int = _SEED + 4):
    """ve_qa_studies: synthetic correctness bench."""
    return _finite_blob(ve_qa_studies.bench_ve_qa_studies(seed))


def bench_vili_qa_studies_family(seed: int = _SEED + 5):
    """vili_qa_studies: synthetic correctness bench."""
    return _finite_blob(vili_qa_studies.bench_vili_qa_studies(seed))
