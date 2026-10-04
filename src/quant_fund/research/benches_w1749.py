"""Wave-1749 bench adapters: slavic-myth-4 canon (SYNTHETIC only)."""

from quant_fund.models import (
    belobog_qa_studies,
    chernobog_qa_studies,
    dazhbog_qa_studies,
    hors_qa_studies,
    semargl_qa_studies,
    stribog_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17490


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_belobog_qa_studies_family(seed: int = _SEED + 0):
    """belobog_qa_studies: synthetic correctness bench."""
    return _finite_blob(belobog_qa_studies.bench_belobog_qa_studies(seed))


def bench_chernobog_qa_studies_family(seed: int = _SEED + 1):
    """chernobog_qa_studies: synthetic correctness bench."""
    return _finite_blob(chernobog_qa_studies.bench_chernobog_qa_studies(seed))


def bench_dazhbog_qa_studies_family(seed: int = _SEED + 2):
    """dazhbog_qa_studies: synthetic correctness bench."""
    return _finite_blob(dazhbog_qa_studies.bench_dazhbog_qa_studies(seed))


def bench_hors_qa_studies_family(seed: int = _SEED + 3):
    """hors_qa_studies: synthetic correctness bench."""
    return _finite_blob(hors_qa_studies.bench_hors_qa_studies(seed))


def bench_semargl_qa_studies_family(seed: int = _SEED + 4):
    """semargl_qa_studies: synthetic correctness bench."""
    return _finite_blob(semargl_qa_studies.bench_semargl_qa_studies(seed))


def bench_stribog_qa_studies_family(seed: int = _SEED + 5):
    """stribog_qa_studies: synthetic correctness bench."""
    return _finite_blob(stribog_qa_studies.bench_stribog_qa_studies(seed))
