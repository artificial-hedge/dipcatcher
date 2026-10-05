"""Wave-1954 bench adapters: goetic-pact canon (SYNTHETIC only)."""

from quant_fund.models import (
    flauros_qa_studies,
    kimaris_qa_studies,
    oriens_qa_studies,
    valac_qa_studies,
    vapula_qa_studies,
    zagan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19540


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_flauros_qa_studies_family(seed: int = _SEED + 0):
    """flauros_qa_studies: synthetic correctness bench."""
    return _finite_blob(flauros_qa_studies.bench_flauros_qa_studies(seed))


def bench_kimaris_qa_studies_family(seed: int = _SEED + 1):
    """kimaris_qa_studies: synthetic correctness bench."""
    return _finite_blob(kimaris_qa_studies.bench_kimaris_qa_studies(seed))


def bench_oriens_qa_studies_family(seed: int = _SEED + 2):
    """oriens_qa_studies: synthetic correctness bench."""
    return _finite_blob(oriens_qa_studies.bench_oriens_qa_studies(seed))


def bench_valac_qa_studies_family(seed: int = _SEED + 3):
    """valac_qa_studies: synthetic correctness bench."""
    return _finite_blob(valac_qa_studies.bench_valac_qa_studies(seed))


def bench_vapula_qa_studies_family(seed: int = _SEED + 4):
    """vapula_qa_studies: synthetic correctness bench."""
    return _finite_blob(vapula_qa_studies.bench_vapula_qa_studies(seed))


def bench_zagan_qa_studies_family(seed: int = _SEED + 5):
    """zagan_qa_studies: synthetic correctness bench."""
    return _finite_blob(zagan_qa_studies.bench_zagan_qa_studies(seed))
