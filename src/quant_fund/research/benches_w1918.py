"""Wave-1918 bench adapters: celtic-demon-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bocan_qa_studies,
    dunter_qa_studies,
    fuath_qa_studies,
    redcap_qa_studies,
    seonaidh_qa_studies,
    wraith_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19180


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bocan_qa_studies_family(seed: int = _SEED + 0):
    """bocan_qa_studies: synthetic correctness bench."""
    return _finite_blob(bocan_qa_studies.bench_bocan_qa_studies(seed))


def bench_dunter_qa_studies_family(seed: int = _SEED + 1):
    """dunter_qa_studies: synthetic correctness bench."""
    return _finite_blob(dunter_qa_studies.bench_dunter_qa_studies(seed))


def bench_fuath_qa_studies_family(seed: int = _SEED + 2):
    """fuath_qa_studies: synthetic correctness bench."""
    return _finite_blob(fuath_qa_studies.bench_fuath_qa_studies(seed))


def bench_redcap_qa_studies_family(seed: int = _SEED + 3):
    """redcap_qa_studies: synthetic correctness bench."""
    return _finite_blob(redcap_qa_studies.bench_redcap_qa_studies(seed))


def bench_seonaidh_qa_studies_family(seed: int = _SEED + 4):
    """seonaidh_qa_studies: synthetic correctness bench."""
    return _finite_blob(seonaidh_qa_studies.bench_seonaidh_qa_studies(seed))


def bench_wraith_qa_studies_family(seed: int = _SEED + 5):
    """wraith_qa_studies: synthetic correctness bench."""
    return _finite_blob(wraith_qa_studies.bench_wraith_qa_studies(seed))
