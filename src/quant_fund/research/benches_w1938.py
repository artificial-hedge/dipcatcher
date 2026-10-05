"""Wave-1938 bench adapters: thai-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    krahang_qa_studies,
    krasue_qa_studies,
    nang_mai_qa_studies,
    phi_am_qa_studies,
    phi_hong_qa_studies,
    phi_pha_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19380


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_krahang_qa_studies_family(seed: int = _SEED + 0):
    """krahang_qa_studies: synthetic correctness bench."""
    return _finite_blob(krahang_qa_studies.bench_krahang_qa_studies(seed))


def bench_krasue_qa_studies_family(seed: int = _SEED + 1):
    """krasue_qa_studies: synthetic correctness bench."""
    return _finite_blob(krasue_qa_studies.bench_krasue_qa_studies(seed))


def bench_nang_mai_qa_studies_family(seed: int = _SEED + 2):
    """nang_mai_qa_studies: synthetic correctness bench."""
    return _finite_blob(nang_mai_qa_studies.bench_nang_mai_qa_studies(seed))


def bench_phi_am_qa_studies_family(seed: int = _SEED + 3):
    """phi_am_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_am_qa_studies.bench_phi_am_qa_studies(seed))


def bench_phi_hong_qa_studies_family(seed: int = _SEED + 4):
    """phi_hong_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_hong_qa_studies.bench_phi_hong_qa_studies(seed))


def bench_phi_pha_qa_studies_family(seed: int = _SEED + 5):
    """phi_pha_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_pha_qa_studies.bench_phi_pha_qa_studies(seed))
