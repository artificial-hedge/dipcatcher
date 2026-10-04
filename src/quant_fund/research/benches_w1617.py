"""Wave-1617 bench adapters: mouse-lemur-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    amber_mountain_qa_studies,
    anosy_qa_studies,
    daraina_qa_studies,
    red_bellied_qa_studies,
    russet_qa_studies,
    white_footed_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16170


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_amber_mountain_qa_studies_family(seed: int = _SEED + 0):
    """amber_mountain_qa_studies: synthetic correctness bench."""
    return _finite_blob(amber_mountain_qa_studies.bench_amber_mountain_qa_studies(seed))


def bench_anosy_qa_studies_family(seed: int = _SEED + 1):
    """anosy_qa_studies: synthetic correctness bench."""
    return _finite_blob(anosy_qa_studies.bench_anosy_qa_studies(seed))


def bench_daraina_qa_studies_family(seed: int = _SEED + 2):
    """daraina_qa_studies: synthetic correctness bench."""
    return _finite_blob(daraina_qa_studies.bench_daraina_qa_studies(seed))


def bench_red_bellied_qa_studies_family(seed: int = _SEED + 3):
    """red_bellied_qa_studies: synthetic correctness bench."""
    return _finite_blob(red_bellied_qa_studies.bench_red_bellied_qa_studies(seed))


def bench_russet_qa_studies_family(seed: int = _SEED + 4):
    """russet_qa_studies: synthetic correctness bench."""
    return _finite_blob(russet_qa_studies.bench_russet_qa_studies(seed))


def bench_white_footed_qa_studies_family(seed: int = _SEED + 5):
    """white_footed_qa_studies: synthetic correctness bench."""
    return _finite_blob(white_footed_qa_studies.bench_white_footed_qa_studies(seed))
