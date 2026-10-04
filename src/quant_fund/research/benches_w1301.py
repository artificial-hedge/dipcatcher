"""Wave-1301 bench adapters: backdoor-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    backdoor_studies,
    clean_label_studies,
    data_poison_studies,
    neural_cleanse_studies,
    spectral_signature_studies,
    trojan_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13010


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_backdoor_studies_family(seed: int = _SEED + 0):
    """backdoor_studies: synthetic correctness bench."""
    return _finite_blob(backdoor_studies.bench_backdoor_studies(seed))


def bench_clean_label_studies_family(seed: int = _SEED + 1):
    """clean_label_studies: synthetic correctness bench."""
    return _finite_blob(clean_label_studies.bench_clean_label_studies(seed))


def bench_data_poison_studies_family(seed: int = _SEED + 2):
    """data_poison_studies: synthetic correctness bench."""
    return _finite_blob(data_poison_studies.bench_data_poison_studies(seed))


def bench_neural_cleanse_studies_family(seed: int = _SEED + 3):
    """neural_cleanse_studies: synthetic correctness bench."""
    return _finite_blob(neural_cleanse_studies.bench_neural_cleanse_studies(seed))


def bench_spectral_signature_studies_family(seed: int = _SEED + 4):
    """spectral_signature_studies: synthetic correctness bench."""
    return _finite_blob(spectral_signature_studies.bench_spectral_signature_studies(seed))


def bench_trojan_studies_family(seed: int = _SEED + 5):
    """trojan_studies: synthetic correctness bench."""
    return _finite_blob(trojan_studies.bench_trojan_studies(seed))
