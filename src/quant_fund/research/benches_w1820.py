"""Wave-1820 bench adapters: chinese-myth-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    changyi2_qa_studies,
    gun2_qa_studies,
    shennong2_qa_studies,
    xihe2_qa_studies,
    yandi2_qa_studies,
    yaoji2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18200


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_changyi2_qa_studies_family(seed: int = _SEED + 0):
    """changyi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(changyi2_qa_studies.bench_changyi2_qa_studies(seed))


def bench_gun2_qa_studies_family(seed: int = _SEED + 1):
    """gun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(gun2_qa_studies.bench_gun2_qa_studies(seed))


def bench_shennong2_qa_studies_family(seed: int = _SEED + 2):
    """shennong2_qa_studies: synthetic correctness bench."""
    return _finite_blob(shennong2_qa_studies.bench_shennong2_qa_studies(seed))


def bench_xihe2_qa_studies_family(seed: int = _SEED + 3):
    """xihe2_qa_studies: synthetic correctness bench."""
    return _finite_blob(xihe2_qa_studies.bench_xihe2_qa_studies(seed))


def bench_yandi2_qa_studies_family(seed: int = _SEED + 4):
    """yandi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(yandi2_qa_studies.bench_yandi2_qa_studies(seed))


def bench_yaoji2_qa_studies_family(seed: int = _SEED + 5):
    """yaoji2_qa_studies: synthetic correctness bench."""
    return _finite_blob(yaoji2_qa_studies.bench_yaoji2_qa_studies(seed))
