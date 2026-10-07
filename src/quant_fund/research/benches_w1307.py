"""Wave-1307 bench adapters: winograd-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    lambada_studies,
    record_studies,
    story_cloze_studies,
    winogender_studies,
    winograd_studies,
    wsc_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13070


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


def bench_lambada_studies_family(seed: int = _SEED + 0):
    """lambada_studies: synthetic correctness bench."""
    return _finite_blob(lambada_studies.bench_lambada_studies(seed))


def bench_record_studies_family(seed: int = _SEED + 1):
    """record_studies: synthetic correctness bench."""
    return _finite_blob(record_studies.bench_record_studies(seed))


def bench_story_cloze_studies_family(seed: int = _SEED + 2):
    """story_cloze_studies: synthetic correctness bench."""
    return _finite_blob(story_cloze_studies.bench_story_cloze_studies(seed))


def bench_winogender_studies_family(seed: int = _SEED + 3):
    """winogender_studies: synthetic correctness bench."""
    return _finite_blob(winogender_studies.bench_winogender_studies(seed))


def bench_winograd_studies_family(seed: int = _SEED + 4):
    """winograd_studies: synthetic correctness bench."""
    return _finite_blob(winograd_studies.bench_winograd_studies(seed))


def bench_wsc_studies_family(seed: int = _SEED + 5):
    """wsc_studies: synthetic correctness bench."""
    return _finite_blob(wsc_studies.bench_wsc_studies(seed))
