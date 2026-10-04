"""Wave-1360 bench adapters: fact-check canon (SYNTHETIC only)."""

from quant_fund.models import (
    bioasq_lite_studies,
    cite_worth_studies,
    climate_fever_studies,
    fever_lite_studies,
    touch_e_studies,
    verdict_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bioasq_lite_studies_family(seed: int = _SEED + 0):
    """bioasq_lite_studies: synthetic correctness bench."""
    return _finite_blob(bioasq_lite_studies.bench_bioasq_lite_studies(seed))


def bench_cite_worth_studies_family(seed: int = _SEED + 1):
    """cite_worth_studies: synthetic correctness bench."""
    return _finite_blob(cite_worth_studies.bench_cite_worth_studies(seed))


def bench_climate_fever_studies_family(seed: int = _SEED + 2):
    """climate_fever_studies: synthetic correctness bench."""
    return _finite_blob(climate_fever_studies.bench_climate_fever_studies(seed))


def bench_fever_lite_studies_family(seed: int = _SEED + 3):
    """fever_lite_studies: synthetic correctness bench."""
    return _finite_blob(fever_lite_studies.bench_fever_lite_studies(seed))


def bench_touch_e_studies_family(seed: int = _SEED + 4):
    """touch_e_studies: synthetic correctness bench."""
    return _finite_blob(touch_e_studies.bench_touch_e_studies(seed))


def bench_verdict_qa_studies_family(seed: int = _SEED + 5):
    """verdict_qa_studies: synthetic correctness bench."""
    return _finite_blob(verdict_qa_studies.bench_verdict_qa_studies(seed))
