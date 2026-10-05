"""Wave-1885 bench adapters: tuareg-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    amenokal_qa_studies,
    ammonion_qa_studies,
    atlas_deity_qa_studies,
    imajeghen_qa_studies,
    melqart_libya_qa_studies,
    tritogeneia_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18850


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amenokal_qa_studies_family(seed: int = _SEED + 0):
    """amenokal_qa_studies: synthetic correctness bench."""
    return _finite_blob(amenokal_qa_studies.bench_amenokal_qa_studies(seed))


def bench_ammonion_qa_studies_family(seed: int = _SEED + 1):
    """ammonion_qa_studies: synthetic correctness bench."""
    return _finite_blob(ammonion_qa_studies.bench_ammonion_qa_studies(seed))


def bench_atlas_deity_qa_studies_family(seed: int = _SEED + 2):
    """atlas_deity_qa_studies: synthetic correctness bench."""
    return _finite_blob(atlas_deity_qa_studies.bench_atlas_deity_qa_studies(seed))


def bench_imajeghen_qa_studies_family(seed: int = _SEED + 3):
    """imajeghen_qa_studies: synthetic correctness bench."""
    return _finite_blob(imajeghen_qa_studies.bench_imajeghen_qa_studies(seed))


def bench_melqart_libya_qa_studies_family(seed: int = _SEED + 4):
    """melqart_libya_qa_studies: synthetic correctness bench."""
    return _finite_blob(melqart_libya_qa_studies.bench_melqart_libya_qa_studies(seed))


def bench_tritogeneia_qa_studies_family(seed: int = _SEED + 5):
    """tritogeneia_qa_studies: synthetic correctness bench."""
    return _finite_blob(tritogeneia_qa_studies.bench_tritogeneia_qa_studies(seed))
