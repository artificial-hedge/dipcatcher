"""Wave-1687 bench adapters: egyptian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    abti_qa_studies,
    akh_qa_studies,
    apep_qa_studies,
    bastet_qa_studies,
    khonsu_qa_studies,
    sobek_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16870


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abti_qa_studies_family(seed: int = _SEED + 0):
    """abti_qa_studies: synthetic correctness bench."""
    return _finite_blob(abti_qa_studies.bench_abti_qa_studies(seed))


def bench_akh_qa_studies_family(seed: int = _SEED + 1):
    """akh_qa_studies: synthetic correctness bench."""
    return _finite_blob(akh_qa_studies.bench_akh_qa_studies(seed))


def bench_apep_qa_studies_family(seed: int = _SEED + 2):
    """apep_qa_studies: synthetic correctness bench."""
    return _finite_blob(apep_qa_studies.bench_apep_qa_studies(seed))


def bench_bastet_qa_studies_family(seed: int = _SEED + 3):
    """bastet_qa_studies: synthetic correctness bench."""
    return _finite_blob(bastet_qa_studies.bench_bastet_qa_studies(seed))


def bench_khonsu_qa_studies_family(seed: int = _SEED + 4):
    """khonsu_qa_studies: synthetic correctness bench."""
    return _finite_blob(khonsu_qa_studies.bench_khonsu_qa_studies(seed))


def bench_sobek_qa_studies_family(seed: int = _SEED + 5):
    """sobek_qa_studies: synthetic correctness bench."""
    return _finite_blob(sobek_qa_studies.bench_sobek_qa_studies(seed))
