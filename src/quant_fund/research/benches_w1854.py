"""Wave-1854 bench adapters: carthaginian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    astarte_punic_qa_studies,
    baal_hammon_qa_studies,
    mekal_qa_studies,
    sid_qa_studies,
    tanit_punic_qa_studies,
    yamm_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18540


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_astarte_punic_qa_studies_family(seed: int = _SEED + 0):
    """astarte_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(astarte_punic_qa_studies.bench_astarte_punic_qa_studies(seed))


def bench_baal_hammon_qa_studies_family(seed: int = _SEED + 1):
    """baal_hammon_qa_studies: synthetic correctness bench."""
    return _finite_blob(baal_hammon_qa_studies.bench_baal_hammon_qa_studies(seed))


def bench_mekal_qa_studies_family(seed: int = _SEED + 2):
    """mekal_qa_studies: synthetic correctness bench."""
    return _finite_blob(mekal_qa_studies.bench_mekal_qa_studies(seed))


def bench_sid_qa_studies_family(seed: int = _SEED + 3):
    """sid_qa_studies: synthetic correctness bench."""
    return _finite_blob(sid_qa_studies.bench_sid_qa_studies(seed))


def bench_tanit_punic_qa_studies_family(seed: int = _SEED + 4):
    """tanit_punic_qa_studies: synthetic correctness bench."""
    return _finite_blob(tanit_punic_qa_studies.bench_tanit_punic_qa_studies(seed))


def bench_yamm_qa_studies_family(seed: int = _SEED + 5):
    """yamm_qa_studies: synthetic correctness bench."""
    return _finite_blob(yamm_qa_studies.bench_yamm_qa_studies(seed))
