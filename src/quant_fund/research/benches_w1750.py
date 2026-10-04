"""Wave-1750 bench adapters: babylonian-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    adad_qa_studies,
    nabu_qa_studies,
    ninlil_qa_studies,
    shamash_qa_studies,
    sin_qa_studies,
    tiamat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17500


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adad_qa_studies_family(seed: int = _SEED + 0):
    """adad_qa_studies: synthetic correctness bench."""
    return _finite_blob(adad_qa_studies.bench_adad_qa_studies(seed))


def bench_nabu_qa_studies_family(seed: int = _SEED + 1):
    """nabu_qa_studies: synthetic correctness bench."""
    return _finite_blob(nabu_qa_studies.bench_nabu_qa_studies(seed))


def bench_ninlil_qa_studies_family(seed: int = _SEED + 2):
    """ninlil_qa_studies: synthetic correctness bench."""
    return _finite_blob(ninlil_qa_studies.bench_ninlil_qa_studies(seed))


def bench_shamash_qa_studies_family(seed: int = _SEED + 3):
    """shamash_qa_studies: synthetic correctness bench."""
    return _finite_blob(shamash_qa_studies.bench_shamash_qa_studies(seed))


def bench_sin_qa_studies_family(seed: int = _SEED + 4):
    """sin_qa_studies: synthetic correctness bench."""
    return _finite_blob(sin_qa_studies.bench_sin_qa_studies(seed))


def bench_tiamat_qa_studies_family(seed: int = _SEED + 5):
    """tiamat_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiamat_qa_studies.bench_tiamat_qa_studies(seed))
