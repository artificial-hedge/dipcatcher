"""Wave-1623 bench adapters: alpine-bird canon (SYNTHETIC only)."""

from quant_fund.models import (
    altai_qa_studies,
    blood_pheasant_qa_studies,
    chukar_qa_studies,
    monal_qa_studies,
    snow_partridge_qa_studies,
    wallcreeper_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16230


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


def bench_altai_qa_studies_family(seed: int = _SEED + 0):
    """altai_qa_studies: synthetic correctness bench."""
    return _finite_blob(altai_qa_studies.bench_altai_qa_studies(seed))


def bench_blood_pheasant_qa_studies_family(seed: int = _SEED + 1):
    """blood_pheasant_qa_studies: synthetic correctness bench."""
    return _finite_blob(blood_pheasant_qa_studies.bench_blood_pheasant_qa_studies(seed))


def bench_chukar_qa_studies_family(seed: int = _SEED + 2):
    """chukar_qa_studies: synthetic correctness bench."""
    return _finite_blob(chukar_qa_studies.bench_chukar_qa_studies(seed))


def bench_monal_qa_studies_family(seed: int = _SEED + 3):
    """monal_qa_studies: synthetic correctness bench."""
    return _finite_blob(monal_qa_studies.bench_monal_qa_studies(seed))


def bench_snow_partridge_qa_studies_family(seed: int = _SEED + 4):
    """snow_partridge_qa_studies: synthetic correctness bench."""
    return _finite_blob(snow_partridge_qa_studies.bench_snow_partridge_qa_studies(seed))


def bench_wallcreeper_qa_studies_family(seed: int = _SEED + 5):
    """wallcreeper_qa_studies: synthetic correctness bench."""
    return _finite_blob(wallcreeper_qa_studies.bench_wallcreeper_qa_studies(seed))
