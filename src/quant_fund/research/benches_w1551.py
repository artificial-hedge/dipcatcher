"""Wave-1551 bench adapters: viper canon (SYNTHETIC only)."""

from quant_fund.models import (
    bushmaster_qa_studies,
    copperhead_qa_studies,
    coral_snake_qa_studies,
    cottonmouth_qa_studies,
    fer_de_lance_qa_studies,
    rattlesnake_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15510


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


def bench_bushmaster_qa_studies_family(seed: int = _SEED + 0):
    """bushmaster_qa_studies: synthetic correctness bench."""
    return _finite_blob(bushmaster_qa_studies.bench_bushmaster_qa_studies(seed))


def bench_copperhead_qa_studies_family(seed: int = _SEED + 1):
    """copperhead_qa_studies: synthetic correctness bench."""
    return _finite_blob(copperhead_qa_studies.bench_copperhead_qa_studies(seed))


def bench_coral_snake_qa_studies_family(seed: int = _SEED + 2):
    """coral_snake_qa_studies: synthetic correctness bench."""
    return _finite_blob(coral_snake_qa_studies.bench_coral_snake_qa_studies(seed))


def bench_cottonmouth_qa_studies_family(seed: int = _SEED + 3):
    """cottonmouth_qa_studies: synthetic correctness bench."""
    return _finite_blob(cottonmouth_qa_studies.bench_cottonmouth_qa_studies(seed))


def bench_fer_de_lance_qa_studies_family(seed: int = _SEED + 4):
    """fer_de_lance_qa_studies: synthetic correctness bench."""
    return _finite_blob(fer_de_lance_qa_studies.bench_fer_de_lance_qa_studies(seed))


def bench_rattlesnake_qa_studies_family(seed: int = _SEED + 5):
    """rattlesnake_qa_studies: synthetic correctness bench."""
    return _finite_blob(rattlesnake_qa_studies.bench_rattlesnake_qa_studies(seed))
