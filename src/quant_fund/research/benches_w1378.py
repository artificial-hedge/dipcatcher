"""Wave-1378 bench adapters: numerical-reasoning canon (SYNTHETIC only)."""

from quant_fund.models import (
    aqua_lite_studies,
    fin_qa_studies,
    math_qa_studies,
    num_glue_studies,
    tab_fact_studies,
    tat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13780


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aqua_lite_studies_family(seed: int = _SEED + 0):
    """aqua_lite_studies: synthetic correctness bench."""
    return _finite_blob(aqua_lite_studies.bench_aqua_lite_studies(seed))


def bench_fin_qa_studies_family(seed: int = _SEED + 1):
    """fin_qa_studies: synthetic correctness bench."""
    return _finite_blob(fin_qa_studies.bench_fin_qa_studies(seed))


def bench_math_qa_studies_family(seed: int = _SEED + 2):
    """math_qa_studies: synthetic correctness bench."""
    return _finite_blob(math_qa_studies.bench_math_qa_studies(seed))


def bench_num_glue_studies_family(seed: int = _SEED + 3):
    """num_glue_studies: synthetic correctness bench."""
    return _finite_blob(num_glue_studies.bench_num_glue_studies(seed))


def bench_tab_fact_studies_family(seed: int = _SEED + 4):
    """tab_fact_studies: synthetic correctness bench."""
    return _finite_blob(tab_fact_studies.bench_tab_fact_studies(seed))


def bench_tat_qa_studies_family(seed: int = _SEED + 5):
    """tat_qa_studies: synthetic correctness bench."""
    return _finite_blob(tat_qa_studies.bench_tat_qa_studies(seed))
