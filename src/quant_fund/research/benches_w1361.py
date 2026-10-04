"""Wave-1361 bench adapters: misinformation canon (SYNTHETIC only)."""

from quant_fund.models import (
    covid_lies_studies,
    evidence_inf_studies,
    hoax_detect_studies,
    liar_lite_studies,
    rumor_eval_studies,
    scidtb_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13610


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_covid_lies_studies_family(seed: int = _SEED + 0):
    """covid_lies_studies: synthetic correctness bench."""
    return _finite_blob(covid_lies_studies.bench_covid_lies_studies(seed))


def bench_evidence_inf_studies_family(seed: int = _SEED + 1):
    """evidence_inf_studies: synthetic correctness bench."""
    return _finite_blob(evidence_inf_studies.bench_evidence_inf_studies(seed))


def bench_hoax_detect_studies_family(seed: int = _SEED + 2):
    """hoax_detect_studies: synthetic correctness bench."""
    return _finite_blob(hoax_detect_studies.bench_hoax_detect_studies(seed))


def bench_liar_lite_studies_family(seed: int = _SEED + 3):
    """liar_lite_studies: synthetic correctness bench."""
    return _finite_blob(liar_lite_studies.bench_liar_lite_studies(seed))


def bench_rumor_eval_studies_family(seed: int = _SEED + 4):
    """rumor_eval_studies: synthetic correctness bench."""
    return _finite_blob(rumor_eval_studies.bench_rumor_eval_studies(seed))


def bench_scidtb_lite_studies_family(seed: int = _SEED + 5):
    """scidtb_lite_studies: synthetic correctness bench."""
    return _finite_blob(scidtb_lite_studies.bench_scidtb_lite_studies(seed))
