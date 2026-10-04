"""Wave-1342 bench adapters: bias-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    fairness_eval_studies,
    gender_bias_studies,
    jigsaw_tox_studies,
    nlp_bias_studies,
    pronoun_bias_studies,
    regard_metric_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13420


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fairness_eval_studies_family(seed: int = _SEED + 0):
    """fairness_eval_studies: synthetic correctness bench."""
    return _finite_blob(fairness_eval_studies.bench_fairness_eval_studies(seed))


def bench_gender_bias_studies_family(seed: int = _SEED + 1):
    """gender_bias_studies: synthetic correctness bench."""
    return _finite_blob(gender_bias_studies.bench_gender_bias_studies(seed))


def bench_jigsaw_tox_studies_family(seed: int = _SEED + 2):
    """jigsaw_tox_studies: synthetic correctness bench."""
    return _finite_blob(jigsaw_tox_studies.bench_jigsaw_tox_studies(seed))


def bench_nlp_bias_studies_family(seed: int = _SEED + 3):
    """nlp_bias_studies: synthetic correctness bench."""
    return _finite_blob(nlp_bias_studies.bench_nlp_bias_studies(seed))


def bench_pronoun_bias_studies_family(seed: int = _SEED + 4):
    """pronoun_bias_studies: synthetic correctness bench."""
    return _finite_blob(pronoun_bias_studies.bench_pronoun_bias_studies(seed))


def bench_regard_metric_studies_family(seed: int = _SEED + 5):
    """regard_metric_studies: synthetic correctness bench."""
    return _finite_blob(regard_metric_studies.bench_regard_metric_studies(seed))
