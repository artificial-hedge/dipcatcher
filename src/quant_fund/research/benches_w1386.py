"""Wave-1386 bench adapters: embodied-game canon (SYNTHETIC only)."""

from quant_fund.models import (
    alfworld_lite_studies,
    babyai_lite_studies,
    crafter_lite_studies,
    jericho_lite_studies,
    scienceworld_studies,
    textworld_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13860


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


def bench_alfworld_lite_studies_family(seed: int = _SEED + 0):
    """alfworld_lite_studies: synthetic correctness bench."""
    return _finite_blob(alfworld_lite_studies.bench_alfworld_lite_studies(seed))


def bench_babyai_lite_studies_family(seed: int = _SEED + 1):
    """babyai_lite_studies: synthetic correctness bench."""
    return _finite_blob(babyai_lite_studies.bench_babyai_lite_studies(seed))


def bench_crafter_lite_studies_family(seed: int = _SEED + 2):
    """crafter_lite_studies: synthetic correctness bench."""
    return _finite_blob(crafter_lite_studies.bench_crafter_lite_studies(seed))


def bench_jericho_lite_studies_family(seed: int = _SEED + 3):
    """jericho_lite_studies: synthetic correctness bench."""
    return _finite_blob(jericho_lite_studies.bench_jericho_lite_studies(seed))


def bench_scienceworld_studies_family(seed: int = _SEED + 4):
    """scienceworld_studies: synthetic correctness bench."""
    return _finite_blob(scienceworld_studies.bench_scienceworld_studies(seed))


def bench_textworld_lite_studies_family(seed: int = _SEED + 5):
    """textworld_lite_studies: synthetic correctness bench."""
    return _finite_blob(textworld_lite_studies.bench_textworld_lite_studies(seed))
