"""Wave-1318 bench adapters: NLP-eval-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cb_studies,
    cola_studies,
    qqp_studies,
    squad_v2_studies,
    sst2_studies,
    wic_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13180


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cb_studies_family(seed: int = _SEED + 0):
    """cb_studies: synthetic correctness bench."""
    return _finite_blob(cb_studies.bench_cb_studies(seed))


def bench_cola_studies_family(seed: int = _SEED + 1):
    """cola_studies: synthetic correctness bench."""
    return _finite_blob(cola_studies.bench_cola_studies(seed))


def bench_qqp_studies_family(seed: int = _SEED + 2):
    """qqp_studies: synthetic correctness bench."""
    return _finite_blob(qqp_studies.bench_qqp_studies(seed))


def bench_squad_v2_studies_family(seed: int = _SEED + 3):
    """squad_v2_studies: synthetic correctness bench."""
    return _finite_blob(squad_v2_studies.bench_squad_v2_studies(seed))


def bench_sst2_studies_family(seed: int = _SEED + 4):
    """sst2_studies: synthetic correctness bench."""
    return _finite_blob(sst2_studies.bench_sst2_studies(seed))


def bench_wic_studies_family(seed: int = _SEED + 5):
    """wic_studies: synthetic correctness bench."""
    return _finite_blob(wic_studies.bench_wic_studies(seed))
