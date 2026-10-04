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
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
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
