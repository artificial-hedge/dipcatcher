"""Wave-1657 bench adapters: french-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    gargoyle_qa_studies,
    guivre_qa_studies,
    melusine_qa_studies,
    quinotaur_qa_studies,
    tarascon_qa_studies,
    tarrasque_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16570


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


def bench_gargoyle_qa_studies_family(seed: int = _SEED + 0):
    """gargoyle_qa_studies: synthetic correctness bench."""
    return _finite_blob(gargoyle_qa_studies.bench_gargoyle_qa_studies(seed))


def bench_guivre_qa_studies_family(seed: int = _SEED + 1):
    """guivre_qa_studies: synthetic correctness bench."""
    return _finite_blob(guivre_qa_studies.bench_guivre_qa_studies(seed))


def bench_melusine_qa_studies_family(seed: int = _SEED + 2):
    """melusine_qa_studies: synthetic correctness bench."""
    return _finite_blob(melusine_qa_studies.bench_melusine_qa_studies(seed))


def bench_quinotaur_qa_studies_family(seed: int = _SEED + 3):
    """quinotaur_qa_studies: synthetic correctness bench."""
    return _finite_blob(quinotaur_qa_studies.bench_quinotaur_qa_studies(seed))


def bench_tarascon_qa_studies_family(seed: int = _SEED + 4):
    """tarascon_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarascon_qa_studies.bench_tarascon_qa_studies(seed))


def bench_tarrasque_qa_studies_family(seed: int = _SEED + 5):
    """tarrasque_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarrasque_qa_studies.bench_tarrasque_qa_studies(seed))
