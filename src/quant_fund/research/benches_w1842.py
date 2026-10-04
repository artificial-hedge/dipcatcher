"""Wave-1842 bench adapters: ammonite-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ammon_qa_studies,
    baalis2_qa_studies,
    el2_qa_studies,
    milcom_qa_studies,
    moloch2_qa_studies,
    sodom2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18420


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ammon_qa_studies_family(seed: int = _SEED + 0):
    """ammon_qa_studies: synthetic correctness bench."""
    return _finite_blob(ammon_qa_studies.bench_ammon_qa_studies(seed))


def bench_baalis2_qa_studies_family(seed: int = _SEED + 1):
    """baalis2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalis2_qa_studies.bench_baalis2_qa_studies(seed))


def bench_el2_qa_studies_family(seed: int = _SEED + 2):
    """el2_qa_studies: synthetic correctness bench."""
    return _finite_blob(el2_qa_studies.bench_el2_qa_studies(seed))


def bench_milcom_qa_studies_family(seed: int = _SEED + 3):
    """milcom_qa_studies: synthetic correctness bench."""
    return _finite_blob(milcom_qa_studies.bench_milcom_qa_studies(seed))


def bench_moloch2_qa_studies_family(seed: int = _SEED + 4):
    """moloch2_qa_studies: synthetic correctness bench."""
    return _finite_blob(moloch2_qa_studies.bench_moloch2_qa_studies(seed))


def bench_sodom2_qa_studies_family(seed: int = _SEED + 5):
    """sodom2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sodom2_qa_studies.bench_sodom2_qa_studies(seed))
