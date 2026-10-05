"""Wave-1913 bench adapters: slavic-demon-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    berstuk_qa_studies,
    chuma_qa_studies,
    koshmar_qa_studies,
    navka_qa_studies,
    perekus_qa_studies,
    rugievit_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19130


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_berstuk_qa_studies_family(seed: int = _SEED + 0):
    """berstuk_qa_studies: synthetic correctness bench."""
    return _finite_blob(berstuk_qa_studies.bench_berstuk_qa_studies(seed))


def bench_chuma_qa_studies_family(seed: int = _SEED + 1):
    """chuma_qa_studies: synthetic correctness bench."""
    return _finite_blob(chuma_qa_studies.bench_chuma_qa_studies(seed))


def bench_koshmar_qa_studies_family(seed: int = _SEED + 2):
    """koshmar_qa_studies: synthetic correctness bench."""
    return _finite_blob(koshmar_qa_studies.bench_koshmar_qa_studies(seed))


def bench_navka_qa_studies_family(seed: int = _SEED + 3):
    """navka_qa_studies: synthetic correctness bench."""
    return _finite_blob(navka_qa_studies.bench_navka_qa_studies(seed))


def bench_perekus_qa_studies_family(seed: int = _SEED + 4):
    """perekus_qa_studies: synthetic correctness bench."""
    return _finite_blob(perekus_qa_studies.bench_perekus_qa_studies(seed))


def bench_rugievit_qa_studies_family(seed: int = _SEED + 5):
    """rugievit_qa_studies: synthetic correctness bench."""
    return _finite_blob(rugievit_qa_studies.bench_rugievit_qa_studies(seed))
