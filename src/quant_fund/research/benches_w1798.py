"""Wave-1798 bench adapters: chinese-myth-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    changxi2_qa_studies,
    fuxi2_qa_studies,
    gonggong2_qa_studies,
    nuwa2_qa_studies,
    shennong2_qa_studies,
    zhurong2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17980


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_changxi2_qa_studies_family(seed: int = _SEED + 0):
    """changxi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(changxi2_qa_studies.bench_changxi2_qa_studies(seed))


def bench_fuxi2_qa_studies_family(seed: int = _SEED + 1):
    """fuxi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(fuxi2_qa_studies.bench_fuxi2_qa_studies(seed))


def bench_gonggong2_qa_studies_family(seed: int = _SEED + 2):
    """gonggong2_qa_studies: synthetic correctness bench."""
    return _finite_blob(gonggong2_qa_studies.bench_gonggong2_qa_studies(seed))


def bench_nuwa2_qa_studies_family(seed: int = _SEED + 3):
    """nuwa2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuwa2_qa_studies.bench_nuwa2_qa_studies(seed))


def bench_shennong2_qa_studies_family(seed: int = _SEED + 4):
    """shennong2_qa_studies: synthetic correctness bench."""
    return _finite_blob(shennong2_qa_studies.bench_shennong2_qa_studies(seed))


def bench_zhurong2_qa_studies_family(seed: int = _SEED + 5):
    """zhurong2_qa_studies: synthetic correctness bench."""
    return _finite_blob(zhurong2_qa_studies.bench_zhurong2_qa_studies(seed))
