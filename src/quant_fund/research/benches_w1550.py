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
