"""Wave-1568 bench adapters: eel canon (SYNTHETIC only)."""

from quant_fund.models import (
    conger_qa_studies,
    garden_eel_qa_studies,
    hagfish_qa_studies,
    lamprey_qa_studies,
    moray_qa_studies,
    ribbon_eel_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15680


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_conger_qa_studies_family(seed: int = _SEED + 0):
    """conger_qa_studies: synthetic correctness bench."""
    return _finite_blob(conger_qa_studies.bench_conger_qa_studies(seed))


def bench_garden_eel_qa_studies_family(seed: int = _SEED + 1):
    """garden_eel_qa_studies: synthetic correctness bench."""
    return _finite_blob(garden_eel_qa_studies.bench_garden_eel_qa_studies(seed))


def bench_hagfish_qa_studies_family(seed: int = _SEED + 2):
    """hagfish_qa_studies: synthetic correctness bench."""
    return _finite_blob(hagfish_qa_studies.bench_hagfish_qa_studies(seed))


def bench_lamprey_qa_studies_family(seed: int = _SEED + 3):
    """lamprey_qa_studies: synthetic correctness bench."""
    return _finite_blob(lamprey_qa_studies.bench_lamprey_qa_studies(seed))


def bench_moray_qa_studies_family(seed: int = _SEED + 4):
    """moray_qa_studies: synthetic correctness bench."""
    return _finite_blob(moray_qa_studies.bench_moray_qa_studies(seed))


def bench_ribbon_eel_qa_studies_family(seed: int = _SEED + 5):
    """ribbon_eel_qa_studies: synthetic correctness bench."""
    return _finite_blob(ribbon_eel_qa_studies.bench_ribbon_eel_qa_studies(seed))
