"""Wave-1316 bench adapters: cue-conflict canon (SYNTHETIC only)."""

from quant_fund.models import (
    backgrounds_studies,
    cue_conflict_studies,
    geirhos_studies,
    imagenet_bg_studies,
    shape_bias_studies,
    texture_bias_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13160


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


def bench_backgrounds_studies_family(seed: int = _SEED + 0):
    """backgrounds_studies: synthetic correctness bench."""
    return _finite_blob(backgrounds_studies.bench_backgrounds_studies(seed))


def bench_cue_conflict_studies_family(seed: int = _SEED + 1):
    """cue_conflict_studies: synthetic correctness bench."""
    return _finite_blob(cue_conflict_studies.bench_cue_conflict_studies(seed))


def bench_geirhos_studies_family(seed: int = _SEED + 2):
    """geirhos_studies: synthetic correctness bench."""
    return _finite_blob(geirhos_studies.bench_geirhos_studies(seed))


def bench_imagenet_bg_studies_family(seed: int = _SEED + 3):
    """imagenet_bg_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_bg_studies.bench_imagenet_bg_studies(seed))


def bench_shape_bias_studies_family(seed: int = _SEED + 4):
    """shape_bias_studies: synthetic correctness bench."""
    return _finite_blob(shape_bias_studies.bench_shape_bias_studies(seed))


def bench_texture_bias_studies_family(seed: int = _SEED + 5):
    """texture_bias_studies: synthetic correctness bench."""
    return _finite_blob(texture_bias_studies.bench_texture_bias_studies(seed))
