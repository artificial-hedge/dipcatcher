"""Wave-1464 bench adapters: monolith canon (SYNTHETIC only)."""

from quant_fund.models import (
    abyss_qa_studies,
    beacon_qa_studies,
    blizzard_qa_studies,
    monolith_qa_studies,
    spire_qa_studies,
    tempest_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14640


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


def bench_abyss_qa_studies_family(seed: int = _SEED + 0):
    """abyss_qa_studies: synthetic correctness bench."""
    return _finite_blob(abyss_qa_studies.bench_abyss_qa_studies(seed))


def bench_beacon_qa_studies_family(seed: int = _SEED + 1):
    """beacon_qa_studies: synthetic correctness bench."""
    return _finite_blob(beacon_qa_studies.bench_beacon_qa_studies(seed))


def bench_blizzard_qa_studies_family(seed: int = _SEED + 2):
    """blizzard_qa_studies: synthetic correctness bench."""
    return _finite_blob(blizzard_qa_studies.bench_blizzard_qa_studies(seed))


def bench_monolith_qa_studies_family(seed: int = _SEED + 3):
    """monolith_qa_studies: synthetic correctness bench."""
    return _finite_blob(monolith_qa_studies.bench_monolith_qa_studies(seed))


def bench_spire_qa_studies_family(seed: int = _SEED + 4):
    """spire_qa_studies: synthetic correctness bench."""
    return _finite_blob(spire_qa_studies.bench_spire_qa_studies(seed))


def bench_tempest_qa_studies_family(seed: int = _SEED + 5):
    """tempest_qa_studies: synthetic correctness bench."""
    return _finite_blob(tempest_qa_studies.bench_tempest_qa_studies(seed))
