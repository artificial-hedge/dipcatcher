"""Wave-1704 bench adapters: babylonian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    adad_qa_studies,
    ashur_qa_studies,
    ishtar_qa_studies,
    nabu_qa_studies,
    shamash_qa_studies,
    sin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17040


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


def bench_ashur_qa_studies_family(seed: int = _SEED + 1):
    """ashur_qa_studies: synthetic correctness bench."""
    return _finite_blob(ashur_qa_studies.bench_ashur_qa_studies(seed))


def bench_ishtar_qa_studies_family(seed: int = _SEED + 2):
    """ishtar_qa_studies: synthetic correctness bench."""
    return _finite_blob(ishtar_qa_studies.bench_ishtar_qa_studies(seed))


def bench_nabu_qa_studies_family(seed: int = _SEED + 3):
    """nabu_qa_studies: synthetic correctness bench."""
    return _finite_blob(nabu_qa_studies.bench_nabu_qa_studies(seed))


def bench_shamash_qa_studies_family(seed: int = _SEED + 4):
    """shamash_qa_studies: synthetic correctness bench."""
    return _finite_blob(shamash_qa_studies.bench_shamash_qa_studies(seed))


def bench_sin_qa_studies_family(seed: int = _SEED + 5):
    """sin_qa_studies: synthetic correctness bench."""
    return _finite_blob(sin_qa_studies.bench_sin_qa_studies(seed))
