"""Wave-1612 bench adapters: highland-grazer canon (SYNTHETIC only)."""

from quant_fund.models import (
    argali_qa_studies,
    bighorn_qa_studies,
    dall_qa_studies,
    llama_qa_studies,
    mouflon_qa_studies,
    urial_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16120


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_argali_qa_studies_family(seed: int = _SEED + 0):
    """argali_qa_studies: synthetic correctness bench."""
    return _finite_blob(argali_qa_studies.bench_argali_qa_studies(seed))


def bench_bighorn_qa_studies_family(seed: int = _SEED + 1):
    """bighorn_qa_studies: synthetic correctness bench."""
    return _finite_blob(bighorn_qa_studies.bench_bighorn_qa_studies(seed))


def bench_dall_qa_studies_family(seed: int = _SEED + 2):
    """dall_qa_studies: synthetic correctness bench."""
    return _finite_blob(dall_qa_studies.bench_dall_qa_studies(seed))


def bench_llama_qa_studies_family(seed: int = _SEED + 3):
    """llama_qa_studies: synthetic correctness bench."""
    return _finite_blob(llama_qa_studies.bench_llama_qa_studies(seed))


def bench_mouflon_qa_studies_family(seed: int = _SEED + 4):
    """mouflon_qa_studies: synthetic correctness bench."""
    return _finite_blob(mouflon_qa_studies.bench_mouflon_qa_studies(seed))


def bench_urial_qa_studies_family(seed: int = _SEED + 5):
    """urial_qa_studies: synthetic correctness bench."""
    return _finite_blob(urial_qa_studies.bench_urial_qa_studies(seed))
