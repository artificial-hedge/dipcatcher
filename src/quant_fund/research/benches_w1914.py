"""Wave-1914 bench adapters: jinn canon (SYNTHETIC only)."""

from quant_fund.models import (
    ghul_qa_studies,
    ifrit_qa_studies,
    jann_qa_studies,
    marid_qa_studies,
    nasnas_qa_studies,
    shaitan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19140


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ghul_qa_studies_family(seed: int = _SEED + 0):
    """ghul_qa_studies: synthetic correctness bench."""
    return _finite_blob(ghul_qa_studies.bench_ghul_qa_studies(seed))


def bench_ifrit_qa_studies_family(seed: int = _SEED + 1):
    """ifrit_qa_studies: synthetic correctness bench."""
    return _finite_blob(ifrit_qa_studies.bench_ifrit_qa_studies(seed))


def bench_jann_qa_studies_family(seed: int = _SEED + 2):
    """jann_qa_studies: synthetic correctness bench."""
    return _finite_blob(jann_qa_studies.bench_jann_qa_studies(seed))


def bench_marid_qa_studies_family(seed: int = _SEED + 3):
    """marid_qa_studies: synthetic correctness bench."""
    return _finite_blob(marid_qa_studies.bench_marid_qa_studies(seed))


def bench_nasnas_qa_studies_family(seed: int = _SEED + 4):
    """nasnas_qa_studies: synthetic correctness bench."""
    return _finite_blob(nasnas_qa_studies.bench_nasnas_qa_studies(seed))


def bench_shaitan_qa_studies_family(seed: int = _SEED + 5):
    """shaitan_qa_studies: synthetic correctness bench."""
    return _finite_blob(shaitan_qa_studies.bench_shaitan_qa_studies(seed))
