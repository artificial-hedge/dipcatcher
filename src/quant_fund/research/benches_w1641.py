"""Wave-1641 bench adapters: mythic-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    bixie_qa_studies,
    fenghuang_qa_studies,
    hundun_qa_studies,
    qiongqi_qa_studies,
    taotie_qa_studies,
    taowu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16410


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bixie_qa_studies_family(seed: int = _SEED + 0):
    """bixie_qa_studies: synthetic correctness bench."""
    return _finite_blob(bixie_qa_studies.bench_bixie_qa_studies(seed))


def bench_fenghuang_qa_studies_family(seed: int = _SEED + 1):
    """fenghuang_qa_studies: synthetic correctness bench."""
    return _finite_blob(fenghuang_qa_studies.bench_fenghuang_qa_studies(seed))


def bench_hundun_qa_studies_family(seed: int = _SEED + 2):
    """hundun_qa_studies: synthetic correctness bench."""
    return _finite_blob(hundun_qa_studies.bench_hundun_qa_studies(seed))


def bench_qiongqi_qa_studies_family(seed: int = _SEED + 3):
    """qiongqi_qa_studies: synthetic correctness bench."""
    return _finite_blob(qiongqi_qa_studies.bench_qiongqi_qa_studies(seed))


def bench_taotie_qa_studies_family(seed: int = _SEED + 4):
    """taotie_qa_studies: synthetic correctness bench."""
    return _finite_blob(taotie_qa_studies.bench_taotie_qa_studies(seed))


def bench_taowu_qa_studies_family(seed: int = _SEED + 5):
    """taowu_qa_studies: synthetic correctness bench."""
    return _finite_blob(taowu_qa_studies.bench_taowu_qa_studies(seed))
