"""Wave-1315 bench adapters: OOD-robustness canon (SYNTHETIC only)."""

from quant_fund.models import (
    imagenet_a_studies,
    imagenet_e_studies,
    imagenet_o_studies,
    imagenet_sketch_studies,
    imagenet_v2_studies,
    stylized_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13150


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


def bench_imagenet_a_studies_family(seed: int = _SEED + 0):
    """imagenet_a_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_a_studies.bench_imagenet_a_studies(seed))


def bench_imagenet_e_studies_family(seed: int = _SEED + 1):
    """imagenet_e_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_e_studies.bench_imagenet_e_studies(seed))


def bench_imagenet_o_studies_family(seed: int = _SEED + 2):
    """imagenet_o_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_o_studies.bench_imagenet_o_studies(seed))


def bench_imagenet_sketch_studies_family(seed: int = _SEED + 3):
    """imagenet_sketch_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_sketch_studies.bench_imagenet_sketch_studies(seed))


def bench_imagenet_v2_studies_family(seed: int = _SEED + 4):
    """imagenet_v2_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_v2_studies.bench_imagenet_v2_studies(seed))


def bench_stylized_studies_family(seed: int = _SEED + 5):
    """stylized_studies: synthetic correctness bench."""
    return _finite_blob(stylized_studies.bench_stylized_studies(seed))
