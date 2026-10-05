"""Wave-1828 bench adapters: finno-ugric-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    ajatar2_qa_studies,
    ilmatar2_qa_studies,
    jumala2_qa_studies,
    metsanhiisi2_qa_studies,
    otso2_qa_studies,
    peikko2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18280


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ajatar2_qa_studies_family(seed: int = _SEED + 0):
    """ajatar2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ajatar2_qa_studies.bench_ajatar2_qa_studies(seed))


def bench_ilmatar2_qa_studies_family(seed: int = _SEED + 1):
    """ilmatar2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ilmatar2_qa_studies.bench_ilmatar2_qa_studies(seed))


def bench_jumala2_qa_studies_family(seed: int = _SEED + 2):
    """jumala2_qa_studies: synthetic correctness bench."""
    return _finite_blob(jumala2_qa_studies.bench_jumala2_qa_studies(seed))


def bench_metsanhiisi2_qa_studies_family(seed: int = _SEED + 3):
    """metsanhiisi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(metsanhiisi2_qa_studies.bench_metsanhiisi2_qa_studies(seed))


def bench_otso2_qa_studies_family(seed: int = _SEED + 4):
    """otso2_qa_studies: synthetic correctness bench."""
    return _finite_blob(otso2_qa_studies.bench_otso2_qa_studies(seed))


def bench_peikko2_qa_studies_family(seed: int = _SEED + 5):
    """peikko2_qa_studies: synthetic correctness bench."""
    return _finite_blob(peikko2_qa_studies.bench_peikko2_qa_studies(seed))
