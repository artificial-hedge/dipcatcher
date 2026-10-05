"""Wave-1915 bench adapters: shedim canon (SYNTHETIC only)."""

from quant_fund.models import (
    dybbuk_qa_studies,
    ibbur_qa_studies,
    lilim_qa_studies,
    mazzik_qa_studies,
    seirim_qa_studies,
    shedim_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19150


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dybbuk_qa_studies_family(seed: int = _SEED + 0):
    """dybbuk_qa_studies: synthetic correctness bench."""
    return _finite_blob(dybbuk_qa_studies.bench_dybbuk_qa_studies(seed))


def bench_ibbur_qa_studies_family(seed: int = _SEED + 1):
    """ibbur_qa_studies: synthetic correctness bench."""
    return _finite_blob(ibbur_qa_studies.bench_ibbur_qa_studies(seed))


def bench_lilim_qa_studies_family(seed: int = _SEED + 2):
    """lilim_qa_studies: synthetic correctness bench."""
    return _finite_blob(lilim_qa_studies.bench_lilim_qa_studies(seed))


def bench_mazzik_qa_studies_family(seed: int = _SEED + 3):
    """mazzik_qa_studies: synthetic correctness bench."""
    return _finite_blob(mazzik_qa_studies.bench_mazzik_qa_studies(seed))


def bench_seirim_qa_studies_family(seed: int = _SEED + 4):
    """seirim_qa_studies: synthetic correctness bench."""
    return _finite_blob(seirim_qa_studies.bench_seirim_qa_studies(seed))


def bench_shedim_qa_studies_family(seed: int = _SEED + 5):
    """shedim_qa_studies: synthetic correctness bench."""
    return _finite_blob(shedim_qa_studies.bench_shedim_qa_studies(seed))
