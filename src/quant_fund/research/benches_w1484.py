"""Wave-1484 bench adapters: marsupial canon (SYNTHETIC only)."""

from quant_fund.models import (
    bandicoot_qa_studies,
    koala_qa_studies,
    numbat_qa_studies,
    quokka_qa_studies,
    wallaby_qa_studies,
    wombat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14840


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


def bench_bandicoot_qa_studies_family(seed: int = _SEED + 0):
    """bandicoot_qa_studies: synthetic correctness bench."""
    return _finite_blob(bandicoot_qa_studies.bench_bandicoot_qa_studies(seed))


def bench_koala_qa_studies_family(seed: int = _SEED + 1):
    """koala_qa_studies: synthetic correctness bench."""
    return _finite_blob(koala_qa_studies.bench_koala_qa_studies(seed))


def bench_numbat_qa_studies_family(seed: int = _SEED + 2):
    """numbat_qa_studies: synthetic correctness bench."""
    return _finite_blob(numbat_qa_studies.bench_numbat_qa_studies(seed))


def bench_quokka_qa_studies_family(seed: int = _SEED + 3):
    """quokka_qa_studies: synthetic correctness bench."""
    return _finite_blob(quokka_qa_studies.bench_quokka_qa_studies(seed))


def bench_wallaby_qa_studies_family(seed: int = _SEED + 4):
    """wallaby_qa_studies: synthetic correctness bench."""
    return _finite_blob(wallaby_qa_studies.bench_wallaby_qa_studies(seed))


def bench_wombat_qa_studies_family(seed: int = _SEED + 5):
    """wombat_qa_studies: synthetic correctness bench."""
    return _finite_blob(wombat_qa_studies.bench_wombat_qa_studies(seed))
