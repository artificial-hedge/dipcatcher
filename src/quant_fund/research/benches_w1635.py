"""Wave-1635 bench adapters: yokai canon (SYNTHETIC only)."""

from quant_fund.models import (
    kappa_qa_studies,
    kitsune_2_qa_studies,
    oni_qa_studies,
    tanuki_2_qa_studies,
    tengu_qa_studies,
    tsukumogami_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16350


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


def bench_kappa_qa_studies_family(seed: int = _SEED + 0):
    """kappa_qa_studies: synthetic correctness bench."""
    return _finite_blob(kappa_qa_studies.bench_kappa_qa_studies(seed))


def bench_kitsune_2_qa_studies_family(seed: int = _SEED + 1):
    """kitsune_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kitsune_2_qa_studies.bench_kitsune_2_qa_studies(seed))


def bench_oni_qa_studies_family(seed: int = _SEED + 2):
    """oni_qa_studies: synthetic correctness bench."""
    return _finite_blob(oni_qa_studies.bench_oni_qa_studies(seed))


def bench_tanuki_2_qa_studies_family(seed: int = _SEED + 3):
    """tanuki_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanuki_2_qa_studies.bench_tanuki_2_qa_studies(seed))


def bench_tengu_qa_studies_family(seed: int = _SEED + 4):
    """tengu_qa_studies: synthetic correctness bench."""
    return _finite_blob(tengu_qa_studies.bench_tengu_qa_studies(seed))


def bench_tsukumogami_qa_studies_family(seed: int = _SEED + 5):
    """tsukumogami_qa_studies: synthetic correctness bench."""
    return _finite_blob(tsukumogami_qa_studies.bench_tsukumogami_qa_studies(seed))
