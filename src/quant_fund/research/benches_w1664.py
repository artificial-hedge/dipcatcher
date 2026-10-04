"""Wave-1664 bench adapters: mythic-menagerie canon (SYNTHETIC only)."""

from quant_fund.models import (
    kraken_qa_studies,
    krampus_qa_studies,
    roc_qa_studies,
    simurgh_qa_studies,
    siren_qa_studies,
    wyvern_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16640


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_kraken_qa_studies_family(seed: int = _SEED + 0):
    """kraken_qa_studies: synthetic correctness bench."""
    return _finite_blob(kraken_qa_studies.bench_kraken_qa_studies(seed))


def bench_krampus_qa_studies_family(seed: int = _SEED + 1):
    """krampus_qa_studies: synthetic correctness bench."""
    return _finite_blob(krampus_qa_studies.bench_krampus_qa_studies(seed))


def bench_roc_qa_studies_family(seed: int = _SEED + 2):
    """roc_qa_studies: synthetic correctness bench."""
    return _finite_blob(roc_qa_studies.bench_roc_qa_studies(seed))


def bench_simurgh_qa_studies_family(seed: int = _SEED + 3):
    """simurgh_qa_studies: synthetic correctness bench."""
    return _finite_blob(simurgh_qa_studies.bench_simurgh_qa_studies(seed))


def bench_siren_qa_studies_family(seed: int = _SEED + 4):
    """siren_qa_studies: synthetic correctness bench."""
    return _finite_blob(siren_qa_studies.bench_siren_qa_studies(seed))


def bench_wyvern_qa_studies_family(seed: int = _SEED + 5):
    """wyvern_qa_studies: synthetic correctness bench."""
    return _finite_blob(wyvern_qa_studies.bench_wyvern_qa_studies(seed))
