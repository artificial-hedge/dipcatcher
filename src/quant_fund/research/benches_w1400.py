"""Wave-1400 bench adapters: KG-QA canon (SYNTHETIC only)."""

from quant_fund.models import (
    cronqa_lite_studies,
    cwq_lite_studies,
    grailqa_studies,
    kgqa_lite_studies,
    pweb_qa_studies,
    qald_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14000


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cronqa_lite_studies_family(seed: int = _SEED + 0):
    """cronqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(cronqa_lite_studies.bench_cronqa_lite_studies(seed))


def bench_cwq_lite_studies_family(seed: int = _SEED + 1):
    """cwq_lite_studies: synthetic correctness bench."""
    return _finite_blob(cwq_lite_studies.bench_cwq_lite_studies(seed))


def bench_grailqa_studies_family(seed: int = _SEED + 2):
    """grailqa_studies: synthetic correctness bench."""
    return _finite_blob(grailqa_studies.bench_grailqa_studies(seed))


def bench_kgqa_lite_studies_family(seed: int = _SEED + 3):
    """kgqa_lite_studies: synthetic correctness bench."""
    return _finite_blob(kgqa_lite_studies.bench_kgqa_lite_studies(seed))


def bench_pweb_qa_studies_family(seed: int = _SEED + 4):
    """pweb_qa_studies: synthetic correctness bench."""
    return _finite_blob(pweb_qa_studies.bench_pweb_qa_studies(seed))


def bench_qald_lite_studies_family(seed: int = _SEED + 5):
    """qald_lite_studies: synthetic correctness bench."""
    return _finite_blob(qald_lite_studies.bench_qald_lite_studies(seed))
