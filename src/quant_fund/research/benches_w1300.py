"""Wave-1300 bench adapters: robustness-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    adversarial_eval_studies,
    autoattack_studies,
    corruption_studies,
    imagenet_c_studies,
    imagenet_r_studies,
    robust_bench_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13000


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_adversarial_eval_studies_family(seed: int = _SEED + 0):
    """adversarial_eval_studies: synthetic correctness bench."""
    return _finite_blob(adversarial_eval_studies.bench_adversarial_eval_studies(seed))


def bench_autoattack_studies_family(seed: int = _SEED + 1):
    """autoattack_studies: synthetic correctness bench."""
    return _finite_blob(autoattack_studies.bench_autoattack_studies(seed))


def bench_corruption_studies_family(seed: int = _SEED + 2):
    """corruption_studies: synthetic correctness bench."""
    return _finite_blob(corruption_studies.bench_corruption_studies(seed))


def bench_imagenet_c_studies_family(seed: int = _SEED + 3):
    """imagenet_c_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_c_studies.bench_imagenet_c_studies(seed))


def bench_imagenet_r_studies_family(seed: int = _SEED + 4):
    """imagenet_r_studies: synthetic correctness bench."""
    return _finite_blob(imagenet_r_studies.bench_imagenet_r_studies(seed))


def bench_robust_bench_studies_family(seed: int = _SEED + 5):
    """robust_bench_studies: synthetic correctness bench."""
    return _finite_blob(robust_bench_studies.bench_robust_bench_studies(seed))
