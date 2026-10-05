"""Wave-1825 bench adapters: baltic-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    aitvaras2_qa_studies,
    kaukas2_qa_studies,
    laime2_qa_studies,
    perkunas2_qa_studies,
    velnias2_qa_studies,
    zemyna2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18250


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aitvaras2_qa_studies_family(seed: int = _SEED + 0):
    """aitvaras2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aitvaras2_qa_studies.bench_aitvaras2_qa_studies(seed))


def bench_kaukas2_qa_studies_family(seed: int = _SEED + 1):
    """kaukas2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaukas2_qa_studies.bench_kaukas2_qa_studies(seed))


def bench_laime2_qa_studies_family(seed: int = _SEED + 2):
    """laime2_qa_studies: synthetic correctness bench."""
    return _finite_blob(laime2_qa_studies.bench_laime2_qa_studies(seed))


def bench_perkunas2_qa_studies_family(seed: int = _SEED + 3):
    """perkunas2_qa_studies: synthetic correctness bench."""
    return _finite_blob(perkunas2_qa_studies.bench_perkunas2_qa_studies(seed))


def bench_velnias2_qa_studies_family(seed: int = _SEED + 4):
    """velnias2_qa_studies: synthetic correctness bench."""
    return _finite_blob(velnias2_qa_studies.bench_velnias2_qa_studies(seed))


def bench_zemyna2_qa_studies_family(seed: int = _SEED + 5):
    """zemyna2_qa_studies: synthetic correctness bench."""
    return _finite_blob(zemyna2_qa_studies.bench_zemyna2_qa_studies(seed))
