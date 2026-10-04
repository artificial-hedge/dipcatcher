"""Wave-1628 bench adapters: exotic-fauna canon (SYNTHETIC only)."""

from quant_fund.models import (
    colugo_qa_studies,
    geoffroy_qa_studies,
    mandarin_qa_studies,
    moray_eel_qa_studies,
    pangolin_2_qa_studies,
    satyr_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16280


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_colugo_qa_studies_family(seed: int = _SEED + 0):
    """colugo_qa_studies: synthetic correctness bench."""
    return _finite_blob(colugo_qa_studies.bench_colugo_qa_studies(seed))


def bench_geoffroy_qa_studies_family(seed: int = _SEED + 1):
    """geoffroy_qa_studies: synthetic correctness bench."""
    return _finite_blob(geoffroy_qa_studies.bench_geoffroy_qa_studies(seed))


def bench_mandarin_qa_studies_family(seed: int = _SEED + 2):
    """mandarin_qa_studies: synthetic correctness bench."""
    return _finite_blob(mandarin_qa_studies.bench_mandarin_qa_studies(seed))


def bench_moray_eel_qa_studies_family(seed: int = _SEED + 3):
    """moray_eel_qa_studies: synthetic correctness bench."""
    return _finite_blob(moray_eel_qa_studies.bench_moray_eel_qa_studies(seed))


def bench_pangolin_2_qa_studies_family(seed: int = _SEED + 4):
    """pangolin_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(pangolin_2_qa_studies.bench_pangolin_2_qa_studies(seed))


def bench_satyr_qa_studies_family(seed: int = _SEED + 5):
    """satyr_qa_studies: synthetic correctness bench."""
    return _finite_blob(satyr_qa_studies.bench_satyr_qa_studies(seed))
