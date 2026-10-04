"""Wave-1582 bench adapters: bat canon (SYNTHETIC only)."""

from quant_fund.models import (
    flying_fox_qa_studies,
    horseshoe_bat_qa_studies,
    leaf_nosed_qa_studies,
    noctule_qa_studies,
    pipistrelle_qa_studies,
    vampire_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15820


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_flying_fox_qa_studies_family(seed: int = _SEED + 0):
    """flying_fox_qa_studies: synthetic correctness bench."""
    return _finite_blob(flying_fox_qa_studies.bench_flying_fox_qa_studies(seed))


def bench_horseshoe_bat_qa_studies_family(seed: int = _SEED + 1):
    """horseshoe_bat_qa_studies: synthetic correctness bench."""
    return _finite_blob(horseshoe_bat_qa_studies.bench_horseshoe_bat_qa_studies(seed))


def bench_leaf_nosed_qa_studies_family(seed: int = _SEED + 2):
    """leaf_nosed_qa_studies: synthetic correctness bench."""
    return _finite_blob(leaf_nosed_qa_studies.bench_leaf_nosed_qa_studies(seed))


def bench_noctule_qa_studies_family(seed: int = _SEED + 3):
    """noctule_qa_studies: synthetic correctness bench."""
    return _finite_blob(noctule_qa_studies.bench_noctule_qa_studies(seed))


def bench_pipistrelle_qa_studies_family(seed: int = _SEED + 4):
    """pipistrelle_qa_studies: synthetic correctness bench."""
    return _finite_blob(pipistrelle_qa_studies.bench_pipistrelle_qa_studies(seed))


def bench_vampire_qa_studies_family(seed: int = _SEED + 5):
    """vampire_qa_studies: synthetic correctness bench."""
    return _finite_blob(vampire_qa_studies.bench_vampire_qa_studies(seed))
