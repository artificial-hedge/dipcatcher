"""Wave-1625 bench adapters: cave-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    blind_salamander_qa_studies,
    cave_shrimp_qa_studies,
    cave_spider_qa_studies,
    cave_swiftlet_qa_studies,
    grotto_salamander_qa_studies,
    proteus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16250


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


def bench_blind_salamander_qa_studies_family(seed: int = _SEED + 0):
    """blind_salamander_qa_studies: synthetic correctness bench."""
    return _finite_blob(blind_salamander_qa_studies.bench_blind_salamander_qa_studies(seed))


def bench_cave_shrimp_qa_studies_family(seed: int = _SEED + 1):
    """cave_shrimp_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_shrimp_qa_studies.bench_cave_shrimp_qa_studies(seed))


def bench_cave_spider_qa_studies_family(seed: int = _SEED + 2):
    """cave_spider_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_spider_qa_studies.bench_cave_spider_qa_studies(seed))


def bench_cave_swiftlet_qa_studies_family(seed: int = _SEED + 3):
    """cave_swiftlet_qa_studies: synthetic correctness bench."""
    return _finite_blob(cave_swiftlet_qa_studies.bench_cave_swiftlet_qa_studies(seed))


def bench_grotto_salamander_qa_studies_family(seed: int = _SEED + 4):
    """grotto_salamander_qa_studies: synthetic correctness bench."""
    return _finite_blob(grotto_salamander_qa_studies.bench_grotto_salamander_qa_studies(seed))


def bench_proteus_qa_studies_family(seed: int = _SEED + 5):
    """proteus_qa_studies: synthetic correctness bench."""
    return _finite_blob(proteus_qa_studies.bench_proteus_qa_studies(seed))
