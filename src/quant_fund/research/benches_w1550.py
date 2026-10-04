"""Wave-1550 bench adapters: lizard canon (SYNTHETIC only)."""

from quant_fund.models import (
    agama_qa_studies,
    chuckwalla_qa_studies,
    frilled_lizard_qa_studies,
    monitor_lizard_qa_studies,
    tegu_qa_studies,
    uromastyx_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15500


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agama_qa_studies_family(seed: int = _SEED + 0):
    """agama_qa_studies: synthetic correctness bench."""
    return _finite_blob(agama_qa_studies.bench_agama_qa_studies(seed))


def bench_chuckwalla_qa_studies_family(seed: int = _SEED + 1):
    """chuckwalla_qa_studies: synthetic correctness bench."""
    return _finite_blob(chuckwalla_qa_studies.bench_chuckwalla_qa_studies(seed))


def bench_frilled_lizard_qa_studies_family(seed: int = _SEED + 2):
    """frilled_lizard_qa_studies: synthetic correctness bench."""
    return _finite_blob(frilled_lizard_qa_studies.bench_frilled_lizard_qa_studies(seed))


def bench_monitor_lizard_qa_studies_family(seed: int = _SEED + 3):
    """monitor_lizard_qa_studies: synthetic correctness bench."""
    return _finite_blob(monitor_lizard_qa_studies.bench_monitor_lizard_qa_studies(seed))


def bench_tegu_qa_studies_family(seed: int = _SEED + 4):
    """tegu_qa_studies: synthetic correctness bench."""
    return _finite_blob(tegu_qa_studies.bench_tegu_qa_studies(seed))


def bench_uromastyx_qa_studies_family(seed: int = _SEED + 5):
    """uromastyx_qa_studies: synthetic correctness bench."""
    return _finite_blob(uromastyx_qa_studies.bench_uromastyx_qa_studies(seed))
