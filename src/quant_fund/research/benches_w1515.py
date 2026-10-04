"""Wave-1515 bench adapters: beetle canon (SYNTHETIC only)."""

from quant_fund.models import (
    click_beetle_qa_studies,
    dung_beetle_qa_studies,
    ground_beetle_qa_studies,
    rhino_beetle_qa_studies,
    stag_beetle_qa_studies,
    tiger_beetle_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15150


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_click_beetle_qa_studies_family(seed: int = _SEED + 0):
    """click_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(click_beetle_qa_studies.bench_click_beetle_qa_studies(seed))


def bench_dung_beetle_qa_studies_family(seed: int = _SEED + 1):
    """dung_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(dung_beetle_qa_studies.bench_dung_beetle_qa_studies(seed))


def bench_ground_beetle_qa_studies_family(seed: int = _SEED + 2):
    """ground_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(ground_beetle_qa_studies.bench_ground_beetle_qa_studies(seed))


def bench_rhino_beetle_qa_studies_family(seed: int = _SEED + 3):
    """rhino_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(rhino_beetle_qa_studies.bench_rhino_beetle_qa_studies(seed))


def bench_stag_beetle_qa_studies_family(seed: int = _SEED + 4):
    """stag_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(stag_beetle_qa_studies.bench_stag_beetle_qa_studies(seed))


def bench_tiger_beetle_qa_studies_family(seed: int = _SEED + 5):
    """tiger_beetle_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiger_beetle_qa_studies.bench_tiger_beetle_qa_studies(seed))
