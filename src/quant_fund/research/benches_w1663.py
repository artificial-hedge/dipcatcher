"""Wave-1663 bench adapters: british-folk canon (SYNTHETIC only)."""

from quant_fund.models import (
    barghest_qa_studies,
    black_dog_qa_studies,
    cat_sith_qa_studies,
    church_grim_qa_studies,
    cwn_annwn_qa_studies,
    grimalkin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16630


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


def bench_barghest_qa_studies_family(seed: int = _SEED + 0):
    """barghest_qa_studies: synthetic correctness bench."""
    return _finite_blob(barghest_qa_studies.bench_barghest_qa_studies(seed))


def bench_black_dog_qa_studies_family(seed: int = _SEED + 1):
    """black_dog_qa_studies: synthetic correctness bench."""
    return _finite_blob(black_dog_qa_studies.bench_black_dog_qa_studies(seed))


def bench_cat_sith_qa_studies_family(seed: int = _SEED + 2):
    """cat_sith_qa_studies: synthetic correctness bench."""
    return _finite_blob(cat_sith_qa_studies.bench_cat_sith_qa_studies(seed))


def bench_church_grim_qa_studies_family(seed: int = _SEED + 3):
    """church_grim_qa_studies: synthetic correctness bench."""
    return _finite_blob(church_grim_qa_studies.bench_church_grim_qa_studies(seed))


def bench_cwn_annwn_qa_studies_family(seed: int = _SEED + 4):
    """cwn_annwn_qa_studies: synthetic correctness bench."""
    return _finite_blob(cwn_annwn_qa_studies.bench_cwn_annwn_qa_studies(seed))


def bench_grimalkin_qa_studies_family(seed: int = _SEED + 5):
    """grimalkin_qa_studies: synthetic correctness bench."""
    return _finite_blob(grimalkin_qa_studies.bench_grimalkin_qa_studies(seed))
