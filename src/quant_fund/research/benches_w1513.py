"""Wave-1513 bench adapters: owl canon (SYNTHETIC only)."""

from quant_fund.models import (
    barnowl_qa_studies,
    barred_owl_qa_studies,
    eagle_owl_qa_studies,
    screech_owl_qa_studies,
    snowy_owl_qa_studies,
    tawny_owl_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15130


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


def bench_barnowl_qa_studies_family(seed: int = _SEED + 0):
    """barnowl_qa_studies: synthetic correctness bench."""
    return _finite_blob(barnowl_qa_studies.bench_barnowl_qa_studies(seed))


def bench_barred_owl_qa_studies_family(seed: int = _SEED + 1):
    """barred_owl_qa_studies: synthetic correctness bench."""
    return _finite_blob(barred_owl_qa_studies.bench_barred_owl_qa_studies(seed))


def bench_eagle_owl_qa_studies_family(seed: int = _SEED + 2):
    """eagle_owl_qa_studies: synthetic correctness bench."""
    return _finite_blob(eagle_owl_qa_studies.bench_eagle_owl_qa_studies(seed))


def bench_screech_owl_qa_studies_family(seed: int = _SEED + 3):
    """screech_owl_qa_studies: synthetic correctness bench."""
    return _finite_blob(screech_owl_qa_studies.bench_screech_owl_qa_studies(seed))


def bench_snowy_owl_qa_studies_family(seed: int = _SEED + 4):
    """snowy_owl_qa_studies: synthetic correctness bench."""
    return _finite_blob(snowy_owl_qa_studies.bench_snowy_owl_qa_studies(seed))


def bench_tawny_owl_qa_studies_family(seed: int = _SEED + 5):
    """tawny_owl_qa_studies: synthetic correctness bench."""
    return _finite_blob(tawny_owl_qa_studies.bench_tawny_owl_qa_studies(seed))
