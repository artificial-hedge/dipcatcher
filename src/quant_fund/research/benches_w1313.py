"""Wave-1313 bench adapters: privacy-attack canon (SYNTHETIC only)."""

from quant_fund.models import (
    abs_scan_studies,
    activation_cluster_studies,
    fine_pruning_studies,
    sleepless_studies,
    strip_defense_studies,
    watermark_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13130


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abs_scan_studies_family(seed: int = _SEED + 0):
    """abs_scan_studies: synthetic correctness bench."""
    return _finite_blob(abs_scan_studies.bench_abs_scan_studies(seed))


def bench_activation_cluster_studies_family(seed: int = _SEED + 1):
    """activation_cluster_studies: synthetic correctness bench."""
    return _finite_blob(activation_cluster_studies.bench_activation_cluster_studies(seed))


def bench_fine_pruning_studies_family(seed: int = _SEED + 2):
    """fine_pruning_studies: synthetic correctness bench."""
    return _finite_blob(fine_pruning_studies.bench_fine_pruning_studies(seed))


def bench_sleepless_studies_family(seed: int = _SEED + 3):
    """sleepless_studies: synthetic correctness bench."""
    return _finite_blob(sleepless_studies.bench_sleepless_studies(seed))


def bench_strip_defense_studies_family(seed: int = _SEED + 4):
    """strip_defense_studies: synthetic correctness bench."""
    return _finite_blob(strip_defense_studies.bench_strip_defense_studies(seed))


def bench_watermark_studies_family(seed: int = _SEED + 5):
    """watermark_studies: synthetic correctness bench."""
    return _finite_blob(watermark_studies.bench_watermark_studies(seed))
