"""Wave-1790 bench adapters: egyptian-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    amun_qa_studies,
    atum_qa_studies,
    khepri_qa_studies,
    mut_qa_studies,
    ptah_qa_studies,
    seth_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17900


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amun_qa_studies_family(seed: int = _SEED + 0):
    """amun_qa_studies: synthetic correctness bench."""
    return _finite_blob(amun_qa_studies.bench_amun_qa_studies(seed))


def bench_atum_qa_studies_family(seed: int = _SEED + 1):
    """atum_qa_studies: synthetic correctness bench."""
    return _finite_blob(atum_qa_studies.bench_atum_qa_studies(seed))


def bench_khepri_qa_studies_family(seed: int = _SEED + 2):
    """khepri_qa_studies: synthetic correctness bench."""
    return _finite_blob(khepri_qa_studies.bench_khepri_qa_studies(seed))


def bench_mut_qa_studies_family(seed: int = _SEED + 3):
    """mut_qa_studies: synthetic correctness bench."""
    return _finite_blob(mut_qa_studies.bench_mut_qa_studies(seed))


def bench_ptah_qa_studies_family(seed: int = _SEED + 4):
    """ptah_qa_studies: synthetic correctness bench."""
    return _finite_blob(ptah_qa_studies.bench_ptah_qa_studies(seed))


def bench_seth_qa_studies_family(seed: int = _SEED + 5):
    """seth_qa_studies: synthetic correctness bench."""
    return _finite_blob(seth_qa_studies.bench_seth_qa_studies(seed))
