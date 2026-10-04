"""Wave-1752 bench adapters: greek-sea canon (SYNTHETIC only)."""

from quant_fund.models import (
    nereus_qa_studies,
    phorcys_qa_studies,
    pontus_qa_studies,
    proteus_qa_studies,
    thaumas_qa_studies,
    triton_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17520


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_nereus_qa_studies_family(seed: int = _SEED + 0):
    """nereus_qa_studies: synthetic correctness bench."""
    return _finite_blob(nereus_qa_studies.bench_nereus_qa_studies(seed))


def bench_phorcys_qa_studies_family(seed: int = _SEED + 1):
    """phorcys_qa_studies: synthetic correctness bench."""
    return _finite_blob(phorcys_qa_studies.bench_phorcys_qa_studies(seed))


def bench_pontus_qa_studies_family(seed: int = _SEED + 2):
    """pontus_qa_studies: synthetic correctness bench."""
    return _finite_blob(pontus_qa_studies.bench_pontus_qa_studies(seed))


def bench_proteus_qa_studies_family(seed: int = _SEED + 3):
    """proteus_qa_studies: synthetic correctness bench."""
    return _finite_blob(proteus_qa_studies.bench_proteus_qa_studies(seed))


def bench_thaumas_qa_studies_family(seed: int = _SEED + 4):
    """thaumas_qa_studies: synthetic correctness bench."""
    return _finite_blob(thaumas_qa_studies.bench_thaumas_qa_studies(seed))


def bench_triton_qa_studies_family(seed: int = _SEED + 5):
    """triton_qa_studies: synthetic correctness bench."""
    return _finite_blob(triton_qa_studies.bench_triton_qa_studies(seed))
