"""Wave-1949 bench adapters: goetic-legion canon (SYNTHETIC only)."""

from quant_fund.models import (
    agares_qa_studies,
    amon_qa_studies,
    barbatos_qa_studies,
    gusion_qa_studies,
    valefor_qa_studies,
    vasago_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19490


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agares_qa_studies_family(seed: int = _SEED + 0):
    """agares_qa_studies: synthetic correctness bench."""
    return _finite_blob(agares_qa_studies.bench_agares_qa_studies(seed))


def bench_amon_qa_studies_family(seed: int = _SEED + 1):
    """amon_qa_studies: synthetic correctness bench."""
    return _finite_blob(amon_qa_studies.bench_amon_qa_studies(seed))


def bench_barbatos_qa_studies_family(seed: int = _SEED + 2):
    """barbatos_qa_studies: synthetic correctness bench."""
    return _finite_blob(barbatos_qa_studies.bench_barbatos_qa_studies(seed))


def bench_gusion_qa_studies_family(seed: int = _SEED + 3):
    """gusion_qa_studies: synthetic correctness bench."""
    return _finite_blob(gusion_qa_studies.bench_gusion_qa_studies(seed))


def bench_valefor_qa_studies_family(seed: int = _SEED + 4):
    """valefor_qa_studies: synthetic correctness bench."""
    return _finite_blob(valefor_qa_studies.bench_valefor_qa_studies(seed))


def bench_vasago_qa_studies_family(seed: int = _SEED + 5):
    """vasago_qa_studies: synthetic correctness bench."""
    return _finite_blob(vasago_qa_studies.bench_vasago_qa_studies(seed))
