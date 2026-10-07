"""Wave-1620 bench adapters: venom-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    boomslang_qa_studies,
    death_adder_qa_studies,
    gaboon_qa_studies,
    inland_taipan_qa_studies,
    saw_scaled_qa_studies,
    sea_krait_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16200


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


def bench_boomslang_qa_studies_family(seed: int = _SEED + 0):
    """boomslang_qa_studies: synthetic correctness bench."""
    return _finite_blob(boomslang_qa_studies.bench_boomslang_qa_studies(seed))


def bench_death_adder_qa_studies_family(seed: int = _SEED + 1):
    """death_adder_qa_studies: synthetic correctness bench."""
    return _finite_blob(death_adder_qa_studies.bench_death_adder_qa_studies(seed))


def bench_gaboon_qa_studies_family(seed: int = _SEED + 2):
    """gaboon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gaboon_qa_studies.bench_gaboon_qa_studies(seed))


def bench_inland_taipan_qa_studies_family(seed: int = _SEED + 3):
    """inland_taipan_qa_studies: synthetic correctness bench."""
    return _finite_blob(inland_taipan_qa_studies.bench_inland_taipan_qa_studies(seed))


def bench_saw_scaled_qa_studies_family(seed: int = _SEED + 4):
    """saw_scaled_qa_studies: synthetic correctness bench."""
    return _finite_blob(saw_scaled_qa_studies.bench_saw_scaled_qa_studies(seed))


def bench_sea_krait_qa_studies_family(seed: int = _SEED + 5):
    """sea_krait_qa_studies: synthetic correctness bench."""
    return _finite_blob(sea_krait_qa_studies.bench_sea_krait_qa_studies(seed))
