"""Wave-1548 bench adapters: cuckoo-turaco canon (SYNTHETIC only)."""

from quant_fund.models import (
    ani_qa_studies,
    coua_qa_studies,
    guira_qa_studies,
    hoatzin_qa_studies,
    malkoha_qa_studies,
    turaco_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15480


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ani_qa_studies_family(seed: int = _SEED + 0):
    """ani_qa_studies: synthetic correctness bench."""
    return _finite_blob(ani_qa_studies.bench_ani_qa_studies(seed))


def bench_coua_qa_studies_family(seed: int = _SEED + 1):
    """coua_qa_studies: synthetic correctness bench."""
    return _finite_blob(coua_qa_studies.bench_coua_qa_studies(seed))


def bench_guira_qa_studies_family(seed: int = _SEED + 2):
    """guira_qa_studies: synthetic correctness bench."""
    return _finite_blob(guira_qa_studies.bench_guira_qa_studies(seed))


def bench_hoatzin_qa_studies_family(seed: int = _SEED + 3):
    """hoatzin_qa_studies: synthetic correctness bench."""
    return _finite_blob(hoatzin_qa_studies.bench_hoatzin_qa_studies(seed))


def bench_malkoha_qa_studies_family(seed: int = _SEED + 4):
    """malkoha_qa_studies: synthetic correctness bench."""
    return _finite_blob(malkoha_qa_studies.bench_malkoha_qa_studies(seed))


def bench_turaco_qa_studies_family(seed: int = _SEED + 5):
    """turaco_qa_studies: synthetic correctness bench."""
    return _finite_blob(turaco_qa_studies.bench_turaco_qa_studies(seed))
