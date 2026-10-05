"""Wave-1939 bench adapters: thai-demon-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    phi_khao_qa_studies,
    phi_pret_qa_studies,
    phi_puay_qa_studies,
    phi_rai_qa_studies,
    phi_taen_qa_studies,
    phi_yuan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19390


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_phi_khao_qa_studies_family(seed: int = _SEED + 0):
    """phi_khao_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_khao_qa_studies.bench_phi_khao_qa_studies(seed))


def bench_phi_pret_qa_studies_family(seed: int = _SEED + 1):
    """phi_pret_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_pret_qa_studies.bench_phi_pret_qa_studies(seed))


def bench_phi_puay_qa_studies_family(seed: int = _SEED + 2):
    """phi_puay_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_puay_qa_studies.bench_phi_puay_qa_studies(seed))


def bench_phi_rai_qa_studies_family(seed: int = _SEED + 3):
    """phi_rai_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_rai_qa_studies.bench_phi_rai_qa_studies(seed))


def bench_phi_taen_qa_studies_family(seed: int = _SEED + 4):
    """phi_taen_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_taen_qa_studies.bench_phi_taen_qa_studies(seed))


def bench_phi_yuan_qa_studies_family(seed: int = _SEED + 5):
    """phi_yuan_qa_studies: synthetic correctness bench."""
    return _finite_blob(phi_yuan_qa_studies.bench_phi_yuan_qa_studies(seed))
