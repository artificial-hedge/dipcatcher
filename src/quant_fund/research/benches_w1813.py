"""Wave-1813 bench adapters: norse-myth-14 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baldr2_qa_studies,
    frigg2_qa_studies,
    idun2_qa_studies,
    njord2_qa_studies,
    sif2_qa_studies,
    tyr2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18130


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baldr2_qa_studies_family(seed: int = _SEED + 0):
    """baldr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baldr2_qa_studies.bench_baldr2_qa_studies(seed))


def bench_frigg2_qa_studies_family(seed: int = _SEED + 1):
    """frigg2_qa_studies: synthetic correctness bench."""
    return _finite_blob(frigg2_qa_studies.bench_frigg2_qa_studies(seed))


def bench_idun2_qa_studies_family(seed: int = _SEED + 2):
    """idun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(idun2_qa_studies.bench_idun2_qa_studies(seed))


def bench_njord2_qa_studies_family(seed: int = _SEED + 3):
    """njord2_qa_studies: synthetic correctness bench."""
    return _finite_blob(njord2_qa_studies.bench_njord2_qa_studies(seed))


def bench_sif2_qa_studies_family(seed: int = _SEED + 4):
    """sif2_qa_studies: synthetic correctness bench."""
    return _finite_blob(sif2_qa_studies.bench_sif2_qa_studies(seed))


def bench_tyr2_qa_studies_family(seed: int = _SEED + 5):
    """tyr2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tyr2_qa_studies.bench_tyr2_qa_studies(seed))
