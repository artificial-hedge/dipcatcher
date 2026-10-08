"""Wave-1430 bench adapters: wildlife canon (SYNTHETIC only)."""

from quant_fund.models import (
    animal_qa_studies,
    bird_qa_studies,
    ecosystem_qa_studies,
    fish_qa_studies,
    habitat_qa_studies,
    insect_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14300


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


def bench_animal_qa_studies_family(seed: int = _SEED + 0):
    """animal_qa_studies: synthetic correctness bench."""
    return _finite_blob(animal_qa_studies.bench_animal_qa_studies(seed))


def bench_bird_qa_studies_family(seed: int = _SEED + 1):
    """bird_qa_studies: synthetic correctness bench."""
    return _finite_blob(bird_qa_studies.bench_bird_qa_studies(seed))


def bench_ecosystem_qa_studies_family(seed: int = _SEED + 2):
    """ecosystem_qa_studies: synthetic correctness bench."""
    return _finite_blob(ecosystem_qa_studies.bench_ecosystem_qa_studies(seed))


def bench_fish_qa_studies_family(seed: int = _SEED + 3):
    """fish_qa_studies: synthetic correctness bench."""
    return _finite_blob(fish_qa_studies.bench_fish_qa_studies(seed))


def bench_habitat_qa_studies_family(seed: int = _SEED + 4):
    """habitat_qa_studies: synthetic correctness bench."""
    return _finite_blob(habitat_qa_studies.bench_habitat_qa_studies(seed))


def bench_insect_qa_studies_family(seed: int = _SEED + 5):
    """insect_qa_studies: synthetic correctness bench."""
    return _finite_blob(insect_qa_studies.bench_insect_qa_studies(seed))
