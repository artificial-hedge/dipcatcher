"""Wave-1779 bench adapters: chinese-myth-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    fuxi_qa_studies,
    huangdi_qa_studies,
    nuwa_qa_studies,
    shennong_qa_studies,
    xihe_qa_studies,
    yandi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17790


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fuxi_qa_studies_family(seed: int = _SEED + 0):
    """fuxi_qa_studies: synthetic correctness bench."""
    return _finite_blob(fuxi_qa_studies.bench_fuxi_qa_studies(seed))


def bench_huangdi_qa_studies_family(seed: int = _SEED + 1):
    """huangdi_qa_studies: synthetic correctness bench."""
    return _finite_blob(huangdi_qa_studies.bench_huangdi_qa_studies(seed))


def bench_nuwa_qa_studies_family(seed: int = _SEED + 2):
    """nuwa_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuwa_qa_studies.bench_nuwa_qa_studies(seed))


def bench_shennong_qa_studies_family(seed: int = _SEED + 3):
    """shennong_qa_studies: synthetic correctness bench."""
    return _finite_blob(shennong_qa_studies.bench_shennong_qa_studies(seed))


def bench_xihe_qa_studies_family(seed: int = _SEED + 4):
    """xihe_qa_studies: synthetic correctness bench."""
    return _finite_blob(xihe_qa_studies.bench_xihe_qa_studies(seed))


def bench_yandi_qa_studies_family(seed: int = _SEED + 5):
    """yandi_qa_studies: synthetic correctness bench."""
    return _finite_blob(yandi_qa_studies.bench_yandi_qa_studies(seed))
