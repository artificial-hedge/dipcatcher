"""Wave-1359 bench adapters: QA-exotics canon (SYNTHETIC only)."""

from quant_fund.models import (
    argu_ana_studies,
    babi_lite_studies,
    curious_qa_studies,
    qasper_lite_studies,
    scifact_lite_studies,
    web_questions_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_argu_ana_studies_family(seed: int = _SEED + 0):
    """argu_ana_studies: synthetic correctness bench."""
    return _finite_blob(argu_ana_studies.bench_argu_ana_studies(seed))


def bench_babi_lite_studies_family(seed: int = _SEED + 1):
    """babi_lite_studies: synthetic correctness bench."""
    return _finite_blob(babi_lite_studies.bench_babi_lite_studies(seed))


def bench_curious_qa_studies_family(seed: int = _SEED + 2):
    """curious_qa_studies: synthetic correctness bench."""
    return _finite_blob(curious_qa_studies.bench_curious_qa_studies(seed))


def bench_qasper_lite_studies_family(seed: int = _SEED + 3):
    """qasper_lite_studies: synthetic correctness bench."""
    return _finite_blob(qasper_lite_studies.bench_qasper_lite_studies(seed))


def bench_scifact_lite_studies_family(seed: int = _SEED + 4):
    """scifact_lite_studies: synthetic correctness bench."""
    return _finite_blob(scifact_lite_studies.bench_scifact_lite_studies(seed))


def bench_web_questions_studies_family(seed: int = _SEED + 5):
    """web_questions_studies: synthetic correctness bench."""
    return _finite_blob(web_questions_studies.bench_web_questions_studies(seed))
