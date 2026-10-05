"""Wave-1878 bench adapters: garamantian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baal_marod_qa_studies,
    bozrum_qa_studies,
    guillyn_qa_studies,
    hammonites_qa_studies,
    weded_qa_studies,
    yamenna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18780


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baal_marod_qa_studies_family(seed: int = _SEED + 0):
    """baal_marod_qa_studies: synthetic correctness bench."""
    return _finite_blob(baal_marod_qa_studies.bench_baal_marod_qa_studies(seed))


def bench_bozrum_qa_studies_family(seed: int = _SEED + 1):
    """bozrum_qa_studies: synthetic correctness bench."""
    return _finite_blob(bozrum_qa_studies.bench_bozrum_qa_studies(seed))


def bench_guillyn_qa_studies_family(seed: int = _SEED + 2):
    """guillyn_qa_studies: synthetic correctness bench."""
    return _finite_blob(guillyn_qa_studies.bench_guillyn_qa_studies(seed))


def bench_hammonites_qa_studies_family(seed: int = _SEED + 3):
    """hammonites_qa_studies: synthetic correctness bench."""
    return _finite_blob(hammonites_qa_studies.bench_hammonites_qa_studies(seed))


def bench_weded_qa_studies_family(seed: int = _SEED + 4):
    """weded_qa_studies: synthetic correctness bench."""
    return _finite_blob(weded_qa_studies.bench_weded_qa_studies(seed))


def bench_yamenna_qa_studies_family(seed: int = _SEED + 5):
    """yamenna_qa_studies: synthetic correctness bench."""
    return _finite_blob(yamenna_qa_studies.bench_yamenna_qa_studies(seed))
