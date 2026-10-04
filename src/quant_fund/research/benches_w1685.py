"""Wave-1685 bench adapters: norse-spirit canon (SYNTHETIC only)."""

from quant_fund.models import (
    ettin_qa_studies,
    fylgja_qa_studies,
    landvaettir_qa_studies,
    nokken_qa_studies,
    seidr_qa_studies,
    vette_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16850


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ettin_qa_studies_family(seed: int = _SEED + 0):
    """ettin_qa_studies: synthetic correctness bench."""
    return _finite_blob(ettin_qa_studies.bench_ettin_qa_studies(seed))


def bench_fylgja_qa_studies_family(seed: int = _SEED + 1):
    """fylgja_qa_studies: synthetic correctness bench."""
    return _finite_blob(fylgja_qa_studies.bench_fylgja_qa_studies(seed))


def bench_landvaettir_qa_studies_family(seed: int = _SEED + 2):
    """landvaettir_qa_studies: synthetic correctness bench."""
    return _finite_blob(landvaettir_qa_studies.bench_landvaettir_qa_studies(seed))


def bench_nokken_qa_studies_family(seed: int = _SEED + 3):
    """nokken_qa_studies: synthetic correctness bench."""
    return _finite_blob(nokken_qa_studies.bench_nokken_qa_studies(seed))


def bench_seidr_qa_studies_family(seed: int = _SEED + 4):
    """seidr_qa_studies: synthetic correctness bench."""
    return _finite_blob(seidr_qa_studies.bench_seidr_qa_studies(seed))


def bench_vette_qa_studies_family(seed: int = _SEED + 5):
    """vette_qa_studies: synthetic correctness bench."""
    return _finite_blob(vette_qa_studies.bench_vette_qa_studies(seed))
