"""Wave-1789 bench adapters: greek-minor canon (SYNTHETIC only)."""

from quant_fund.models import (
    eris_qa_studies,
    ganymede_qa_studies,
    hebe_qa_studies,
    hermes_qa_studies,
    momus_qa_studies,
    oneiros_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17890


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_eris_qa_studies_family(seed: int = _SEED + 0):
    """eris_qa_studies: synthetic correctness bench."""
    return _finite_blob(eris_qa_studies.bench_eris_qa_studies(seed))


def bench_ganymede_qa_studies_family(seed: int = _SEED + 1):
    """ganymede_qa_studies: synthetic correctness bench."""
    return _finite_blob(ganymede_qa_studies.bench_ganymede_qa_studies(seed))


def bench_hebe_qa_studies_family(seed: int = _SEED + 2):
    """hebe_qa_studies: synthetic correctness bench."""
    return _finite_blob(hebe_qa_studies.bench_hebe_qa_studies(seed))


def bench_hermes_qa_studies_family(seed: int = _SEED + 3):
    """hermes_qa_studies: synthetic correctness bench."""
    return _finite_blob(hermes_qa_studies.bench_hermes_qa_studies(seed))


def bench_momus_qa_studies_family(seed: int = _SEED + 4):
    """momus_qa_studies: synthetic correctness bench."""
    return _finite_blob(momus_qa_studies.bench_momus_qa_studies(seed))


def bench_oneiros_qa_studies_family(seed: int = _SEED + 5):
    """oneiros_qa_studies: synthetic correctness bench."""
    return _finite_blob(oneiros_qa_studies.bench_oneiros_qa_studies(seed))
