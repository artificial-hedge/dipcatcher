"""Wave-1351 bench adapters: ethics-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    ethic_jiminy_studies,
    moral_exc_studies,
    moral_found_studies,
    principlism_toy_studies,
    scruples_lite_studies,
    virtue_ethics_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13510


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


def bench_ethic_jiminy_studies_family(seed: int = _SEED + 0):
    """ethic_jiminy_studies: synthetic correctness bench."""
    return _finite_blob(ethic_jiminy_studies.bench_ethic_jiminy_studies(seed))


def bench_moral_exc_studies_family(seed: int = _SEED + 1):
    """moral_exc_studies: synthetic correctness bench."""
    return _finite_blob(moral_exc_studies.bench_moral_exc_studies(seed))


def bench_moral_found_studies_family(seed: int = _SEED + 2):
    """moral_found_studies: synthetic correctness bench."""
    return _finite_blob(moral_found_studies.bench_moral_found_studies(seed))


def bench_principlism_toy_studies_family(seed: int = _SEED + 3):
    """principlism_toy_studies: synthetic correctness bench."""
    return _finite_blob(principlism_toy_studies.bench_principlism_toy_studies(seed))


def bench_scruples_lite_studies_family(seed: int = _SEED + 4):
    """scruples_lite_studies: synthetic correctness bench."""
    return _finite_blob(scruples_lite_studies.bench_scruples_lite_studies(seed))


def bench_virtue_ethics_studies_family(seed: int = _SEED + 5):
    """virtue_ethics_studies: synthetic correctness bench."""
    return _finite_blob(virtue_ethics_studies.bench_virtue_ethics_studies(seed))
