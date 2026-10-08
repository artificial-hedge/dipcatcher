"""Wave-1341 bench adapters: bias-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    bold_eval_studies,
    crow_s_pairs_studies,
    hate_speech_eval_studies,
    holo_bias_studies,
    real_toxicity_studies,
    stereo_set_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13410


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bold_eval_studies_family(seed: int = _SEED + 0):
    """bold_eval_studies: synthetic correctness bench."""
    return _finite_blob(bold_eval_studies.bench_bold_eval_studies(seed))


def bench_crow_s_pairs_studies_family(seed: int = _SEED + 1):
    """crow_s_pairs_studies: synthetic correctness bench."""
    return _finite_blob(crow_s_pairs_studies.bench_crow_s_pairs_studies(seed))


def bench_hate_speech_eval_studies_family(seed: int = _SEED + 2):
    """hate_speech_eval_studies: synthetic correctness bench."""
    return _finite_blob(hate_speech_eval_studies.bench_hate_speech_eval_studies(seed))


def bench_holo_bias_studies_family(seed: int = _SEED + 3):
    """holo_bias_studies: synthetic correctness bench."""
    return _finite_blob(holo_bias_studies.bench_holo_bias_studies(seed))


def bench_real_toxicity_studies_family(seed: int = _SEED + 4):
    """real_toxicity_studies: synthetic correctness bench."""
    return _finite_blob(real_toxicity_studies.bench_real_toxicity_studies(seed))


def bench_stereo_set_studies_family(seed: int = _SEED + 5):
    """stereo_set_studies: synthetic correctness bench."""
    return _finite_blob(stereo_set_studies.bench_stereo_set_studies(seed))
