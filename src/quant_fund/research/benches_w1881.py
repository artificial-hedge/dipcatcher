"""Wave-1881 bench adapters: numidian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    juba_qa_studies,
    jugurtha_qa_studies,
    massinissa_qa_studies,
    micipsa_qa_studies,
    naravas_qa_studies,
    syphax_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18810


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_juba_qa_studies_family(seed: int = _SEED + 0):
    """juba_qa_studies: synthetic correctness bench."""
    return _finite_blob(juba_qa_studies.bench_juba_qa_studies(seed))


def bench_jugurtha_qa_studies_family(seed: int = _SEED + 1):
    """jugurtha_qa_studies: synthetic correctness bench."""
    return _finite_blob(jugurtha_qa_studies.bench_jugurtha_qa_studies(seed))


def bench_massinissa_qa_studies_family(seed: int = _SEED + 2):
    """massinissa_qa_studies: synthetic correctness bench."""
    return _finite_blob(massinissa_qa_studies.bench_massinissa_qa_studies(seed))


def bench_micipsa_qa_studies_family(seed: int = _SEED + 3):
    """micipsa_qa_studies: synthetic correctness bench."""
    return _finite_blob(micipsa_qa_studies.bench_micipsa_qa_studies(seed))


def bench_naravas_qa_studies_family(seed: int = _SEED + 4):
    """naravas_qa_studies: synthetic correctness bench."""
    return _finite_blob(naravas_qa_studies.bench_naravas_qa_studies(seed))


def bench_syphax_qa_studies_family(seed: int = _SEED + 5):
    """syphax_qa_studies: synthetic correctness bench."""
    return _finite_blob(syphax_qa_studies.bench_syphax_qa_studies(seed))
