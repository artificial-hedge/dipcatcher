"""Wave-1520 bench adapters: fungi canon (SYNTHETIC only)."""

from quant_fund.models import (
    agaric_qa_studies,
    bolete_qa_studies,
    chanterelle_qa_studies,
    inkcap_qa_studies,
    morel_qa_studies,
    puffball_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15200


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_agaric_qa_studies_family(seed: int = _SEED + 0):
    """agaric_qa_studies: synthetic correctness bench."""
    return _finite_blob(agaric_qa_studies.bench_agaric_qa_studies(seed))


def bench_bolete_qa_studies_family(seed: int = _SEED + 1):
    """bolete_qa_studies: synthetic correctness bench."""
    return _finite_blob(bolete_qa_studies.bench_bolete_qa_studies(seed))


def bench_chanterelle_qa_studies_family(seed: int = _SEED + 2):
    """chanterelle_qa_studies: synthetic correctness bench."""
    return _finite_blob(chanterelle_qa_studies.bench_chanterelle_qa_studies(seed))


def bench_inkcap_qa_studies_family(seed: int = _SEED + 3):
    """inkcap_qa_studies: synthetic correctness bench."""
    return _finite_blob(inkcap_qa_studies.bench_inkcap_qa_studies(seed))


def bench_morel_qa_studies_family(seed: int = _SEED + 4):
    """morel_qa_studies: synthetic correctness bench."""
    return _finite_blob(morel_qa_studies.bench_morel_qa_studies(seed))


def bench_puffball_qa_studies_family(seed: int = _SEED + 5):
    """puffball_qa_studies: synthetic correctness bench."""
    return _finite_blob(puffball_qa_studies.bench_puffball_qa_studies(seed))
