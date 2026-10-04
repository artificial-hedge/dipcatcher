"""Wave-1784 bench adapters: greek-myth-9 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ares_qa_studies,
    hades_qa_studies,
    hephaestus_qa_studies,
    hestia_qa_studies,
    poseidon_qa_studies,
    zeus_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17840


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ares_qa_studies_family(seed: int = _SEED + 0):
    """ares_qa_studies: synthetic correctness bench."""
    return _finite_blob(ares_qa_studies.bench_ares_qa_studies(seed))


def bench_hades_qa_studies_family(seed: int = _SEED + 1):
    """hades_qa_studies: synthetic correctness bench."""
    return _finite_blob(hades_qa_studies.bench_hades_qa_studies(seed))


def bench_hephaestus_qa_studies_family(seed: int = _SEED + 2):
    """hephaestus_qa_studies: synthetic correctness bench."""
    return _finite_blob(hephaestus_qa_studies.bench_hephaestus_qa_studies(seed))


def bench_hestia_qa_studies_family(seed: int = _SEED + 3):
    """hestia_qa_studies: synthetic correctness bench."""
    return _finite_blob(hestia_qa_studies.bench_hestia_qa_studies(seed))


def bench_poseidon_qa_studies_family(seed: int = _SEED + 4):
    """poseidon_qa_studies: synthetic correctness bench."""
    return _finite_blob(poseidon_qa_studies.bench_poseidon_qa_studies(seed))


def bench_zeus_qa_studies_family(seed: int = _SEED + 5):
    """zeus_qa_studies: synthetic correctness bench."""
    return _finite_blob(zeus_qa_studies.bench_zeus_qa_studies(seed))
