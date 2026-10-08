"""Wave-1618 bench adapters: abyssal canon (SYNTHETIC only)."""

from quant_fund.models import (
    anglerfish_qa_studies,
    bristlemouth_qa_studies,
    grenadier_qa_studies,
    hatchetfish_qa_studies,
    lanternfish_qa_studies,
    viperfish_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16180


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


def bench_anglerfish_qa_studies_family(seed: int = _SEED + 0):
    """anglerfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(anglerfish_qa_studies.bench_anglerfish_qa_studies(seed))


def bench_bristlemouth_qa_studies_family(seed: int = _SEED + 1):
    """bristlemouth_qa_studies: synthetic correctness bench."""
    return _finite_blob(bristlemouth_qa_studies.bench_bristlemouth_qa_studies(seed))


def bench_grenadier_qa_studies_family(seed: int = _SEED + 2):
    """grenadier_qa_studies: synthetic correctness bench."""
    return _finite_blob(grenadier_qa_studies.bench_grenadier_qa_studies(seed))


def bench_hatchetfish_qa_studies_family(seed: int = _SEED + 3):
    """hatchetfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(hatchetfish_qa_studies.bench_hatchetfish_qa_studies(seed))


def bench_lanternfish_qa_studies_family(seed: int = _SEED + 4):
    """lanternfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(lanternfish_qa_studies.bench_lanternfish_qa_studies(seed))


def bench_viperfish_qa_studies_family(seed: int = _SEED + 5):
    """viperfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(viperfish_qa_studies.bench_viperfish_qa_studies(seed))
