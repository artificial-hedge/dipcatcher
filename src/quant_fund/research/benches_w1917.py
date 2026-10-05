"""Wave-1917 bench adapters: celtic-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    baobhan_sith_qa_studies,
    bean_nighe_qa_studies,
    boggart_qa_studies,
    fear_durach_qa_studies,
    glaistig_qa_studies,
    sluagh_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19170


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baobhan_sith_qa_studies_family(seed: int = _SEED + 0):
    """baobhan_sith_qa_studies: synthetic correctness bench."""
    return _finite_blob(baobhan_sith_qa_studies.bench_baobhan_sith_qa_studies(seed))


def bench_bean_nighe_qa_studies_family(seed: int = _SEED + 1):
    """bean_nighe_qa_studies: synthetic correctness bench."""
    return _finite_blob(bean_nighe_qa_studies.bench_bean_nighe_qa_studies(seed))


def bench_boggart_qa_studies_family(seed: int = _SEED + 2):
    """boggart_qa_studies: synthetic correctness bench."""
    return _finite_blob(boggart_qa_studies.bench_boggart_qa_studies(seed))


def bench_fear_durach_qa_studies_family(seed: int = _SEED + 3):
    """fear_durach_qa_studies: synthetic correctness bench."""
    return _finite_blob(fear_durach_qa_studies.bench_fear_durach_qa_studies(seed))


def bench_glaistig_qa_studies_family(seed: int = _SEED + 4):
    """glaistig_qa_studies: synthetic correctness bench."""
    return _finite_blob(glaistig_qa_studies.bench_glaistig_qa_studies(seed))


def bench_sluagh_qa_studies_family(seed: int = _SEED + 5):
    """sluagh_qa_studies: synthetic correctness bench."""
    return _finite_blob(sluagh_qa_studies.bench_sluagh_qa_studies(seed))
