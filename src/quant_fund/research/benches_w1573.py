"""Wave-1573 bench adapters: desert canon (SYNTHETIC only)."""

from quant_fund.models import (
    addax_qa_studies,
    fennec_qa_studies,
    jerboa_qa_studies,
    meerkat_qa_studies,
    onager_qa_studies,
    pangolin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15730


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_addax_qa_studies_family(seed: int = _SEED + 0):
    """addax_qa_studies: synthetic correctness bench."""
    return _finite_blob(addax_qa_studies.bench_addax_qa_studies(seed))


def bench_fennec_qa_studies_family(seed: int = _SEED + 1):
    """fennec_qa_studies: synthetic correctness bench."""
    return _finite_blob(fennec_qa_studies.bench_fennec_qa_studies(seed))


def bench_jerboa_qa_studies_family(seed: int = _SEED + 2):
    """jerboa_qa_studies: synthetic correctness bench."""
    return _finite_blob(jerboa_qa_studies.bench_jerboa_qa_studies(seed))


def bench_meerkat_qa_studies_family(seed: int = _SEED + 3):
    """meerkat_qa_studies: synthetic correctness bench."""
    return _finite_blob(meerkat_qa_studies.bench_meerkat_qa_studies(seed))


def bench_onager_qa_studies_family(seed: int = _SEED + 4):
    """onager_qa_studies: synthetic correctness bench."""
    return _finite_blob(onager_qa_studies.bench_onager_qa_studies(seed))


def bench_pangolin_qa_studies_family(seed: int = _SEED + 5):
    """pangolin_qa_studies: synthetic correctness bench."""
    return _finite_blob(pangolin_qa_studies.bench_pangolin_qa_studies(seed))
