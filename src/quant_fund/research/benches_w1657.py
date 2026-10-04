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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
