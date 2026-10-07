"""Wave-1479 bench adapters: neotropical canon (SYNTHETIC only)."""

from quant_fund.models import (
    agouti_qa_studies,
    armadillo_qa_studies,
    capybara_qa_studies,
    coati_qa_studies,
    peccary_qa_studies,
    tapir_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14790


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


def bench_agouti_qa_studies_family(seed: int = _SEED + 0):
    """agouti_qa_studies: synthetic correctness bench."""
    return _finite_blob(agouti_qa_studies.bench_agouti_qa_studies(seed))


def bench_armadillo_qa_studies_family(seed: int = _SEED + 1):
    """armadillo_qa_studies: synthetic correctness bench."""
    return _finite_blob(armadillo_qa_studies.bench_armadillo_qa_studies(seed))


def bench_capybara_qa_studies_family(seed: int = _SEED + 2):
    """capybara_qa_studies: synthetic correctness bench."""
    return _finite_blob(capybara_qa_studies.bench_capybara_qa_studies(seed))


def bench_coati_qa_studies_family(seed: int = _SEED + 3):
    """coati_qa_studies: synthetic correctness bench."""
    return _finite_blob(coati_qa_studies.bench_coati_qa_studies(seed))


def bench_peccary_qa_studies_family(seed: int = _SEED + 4):
    """peccary_qa_studies: synthetic correctness bench."""
    return _finite_blob(peccary_qa_studies.bench_peccary_qa_studies(seed))


def bench_tapir_qa_studies_family(seed: int = _SEED + 5):
    """tapir_qa_studies: synthetic correctness bench."""
    return _finite_blob(tapir_qa_studies.bench_tapir_qa_studies(seed))
