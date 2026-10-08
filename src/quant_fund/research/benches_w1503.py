"""Wave-1503 bench adapters: serpent-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    garter_qa_studies,
    keelback_qa_studies,
    kingsnake_qa_studies,
    mockviper_qa_studies,
    racer_qa_studies,
    sidewinder_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15030


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


def bench_garter_qa_studies_family(seed: int = _SEED + 0):
    """garter_qa_studies: synthetic correctness bench."""
    return _finite_blob(garter_qa_studies.bench_garter_qa_studies(seed))


def bench_keelback_qa_studies_family(seed: int = _SEED + 1):
    """keelback_qa_studies: synthetic correctness bench."""
    return _finite_blob(keelback_qa_studies.bench_keelback_qa_studies(seed))


def bench_kingsnake_qa_studies_family(seed: int = _SEED + 2):
    """kingsnake_qa_studies: synthetic correctness bench."""
    return _finite_blob(kingsnake_qa_studies.bench_kingsnake_qa_studies(seed))


def bench_mockviper_qa_studies_family(seed: int = _SEED + 3):
    """mockviper_qa_studies: synthetic correctness bench."""
    return _finite_blob(mockviper_qa_studies.bench_mockviper_qa_studies(seed))


def bench_racer_qa_studies_family(seed: int = _SEED + 4):
    """racer_qa_studies: synthetic correctness bench."""
    return _finite_blob(racer_qa_studies.bench_racer_qa_studies(seed))


def bench_sidewinder_qa_studies_family(seed: int = _SEED + 5):
    """sidewinder_qa_studies: synthetic correctness bench."""
    return _finite_blob(sidewinder_qa_studies.bench_sidewinder_qa_studies(seed))
