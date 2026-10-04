"""Wave-1260 bench adapters: clinical-research-methods canon (SYNTHETIC only)."""

from quant_fund.models import (
    adaptive_trial_studies,
    clinical_trial_studies,
    comparative_effectiveness_studies,
    meta_analysis_studies,
    outcomes_research_studies,
    rwe_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 12600


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adaptive_trial_studies_family(seed: int = _SEED + 0):
    """adaptive_trial_studies: synthetic correctness bench."""
    return _finite_blob(adaptive_trial_studies.bench_adaptive_trial_studies(seed))


def bench_clinical_trial_studies_family(seed: int = _SEED + 1):
    """clinical_trial_studies: synthetic correctness bench."""
    return _finite_blob(clinical_trial_studies.bench_clinical_trial_studies(seed))


def bench_comparative_effectiveness_studies_family(seed: int = _SEED + 2):
    """comparative_effectiveness_studies: synthetic correctness bench."""
    return _finite_blob(
        comparative_effectiveness_studies.bench_comparative_effectiveness_studies(seed)
    )


def bench_meta_analysis_studies_family(seed: int = _SEED + 3):
    """meta_analysis_studies: synthetic correctness bench."""
    return _finite_blob(meta_analysis_studies.bench_meta_analysis_studies(seed))


def bench_outcomes_research_studies_family(seed: int = _SEED + 4):
    """outcomes_research_studies: synthetic correctness bench."""
    return _finite_blob(outcomes_research_studies.bench_outcomes_research_studies(seed))


def bench_rwe_studies_family(seed: int = _SEED + 5):
    """rwe_studies: synthetic correctness bench."""
    return _finite_blob(rwe_studies.bench_rwe_studies(seed))
