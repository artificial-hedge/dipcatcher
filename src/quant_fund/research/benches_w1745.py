"""Wave-1745 bench adapters: aboriginal-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    altjira_qa_studies,
    bunyip_qa_studies,
    mimis_qa_studies,
    rainbow_serpent_qa_studies,
    wandjina_qa_studies,
    yowie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17450


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_altjira_qa_studies_family(seed: int = _SEED + 0):
    """altjira_qa_studies: synthetic correctness bench."""
    return _finite_blob(altjira_qa_studies.bench_altjira_qa_studies(seed))


def bench_bunyip_qa_studies_family(seed: int = _SEED + 1):
    """bunyip_qa_studies: synthetic correctness bench."""
    return _finite_blob(bunyip_qa_studies.bench_bunyip_qa_studies(seed))


def bench_mimis_qa_studies_family(seed: int = _SEED + 2):
    """mimis_qa_studies: synthetic correctness bench."""
    return _finite_blob(mimis_qa_studies.bench_mimis_qa_studies(seed))


def bench_rainbow_serpent_qa_studies_family(seed: int = _SEED + 3):
    """rainbow_serpent_qa_studies: synthetic correctness bench."""
    return _finite_blob(rainbow_serpent_qa_studies.bench_rainbow_serpent_qa_studies(seed))


def bench_wandjina_qa_studies_family(seed: int = _SEED + 4):
    """wandjina_qa_studies: synthetic correctness bench."""
    return _finite_blob(wandjina_qa_studies.bench_wandjina_qa_studies(seed))


def bench_yowie_qa_studies_family(seed: int = _SEED + 5):
    """yowie_qa_studies: synthetic correctness bench."""
    return _finite_blob(yowie_qa_studies.bench_yowie_qa_studies(seed))
