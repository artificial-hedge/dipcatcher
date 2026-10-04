"""Wave-1714 bench adapters: scythian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    argimpasa_qa_studies,
    arimasp_qa_studies,
    papaios_qa_studies,
    tabiti_qa_studies,
    tavrita_qa_studies,
    thagimasadas_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17140


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_argimpasa_qa_studies_family(seed: int = _SEED + 0):
    """argimpasa_qa_studies: synthetic correctness bench."""
    return _finite_blob(argimpasa_qa_studies.bench_argimpasa_qa_studies(seed))


def bench_arimasp_qa_studies_family(seed: int = _SEED + 1):
    """arimasp_qa_studies: synthetic correctness bench."""
    return _finite_blob(arimasp_qa_studies.bench_arimasp_qa_studies(seed))


def bench_papaios_qa_studies_family(seed: int = _SEED + 2):
    """papaios_qa_studies: synthetic correctness bench."""
    return _finite_blob(papaios_qa_studies.bench_papaios_qa_studies(seed))


def bench_tabiti_qa_studies_family(seed: int = _SEED + 3):
    """tabiti_qa_studies: synthetic correctness bench."""
    return _finite_blob(tabiti_qa_studies.bench_tabiti_qa_studies(seed))


def bench_tavrita_qa_studies_family(seed: int = _SEED + 4):
    """tavrita_qa_studies: synthetic correctness bench."""
    return _finite_blob(tavrita_qa_studies.bench_tavrita_qa_studies(seed))


def bench_thagimasadas_qa_studies_family(seed: int = _SEED + 5):
    """thagimasadas_qa_studies: synthetic correctness bench."""
    return _finite_blob(thagimasadas_qa_studies.bench_thagimasadas_qa_studies(seed))
