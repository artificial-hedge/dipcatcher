"""Wave-1333 bench adapters: multilingual-eval canon (SYNTHETIC only)."""

from quant_fund.models import (
    fava_studies,
    polyglo_tox_studies,
    regard_eval_studies,
    unqover_studies,
    vlur_studies,
    xlsum_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13330


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_fava_studies_family(seed: int = _SEED + 0):
    """fava_studies: synthetic correctness bench."""
    return _finite_blob(fava_studies.bench_fava_studies(seed))


def bench_polyglo_tox_studies_family(seed: int = _SEED + 1):
    """polyglo_tox_studies: synthetic correctness bench."""
    return _finite_blob(polyglo_tox_studies.bench_polyglo_tox_studies(seed))


def bench_regard_eval_studies_family(seed: int = _SEED + 2):
    """regard_eval_studies: synthetic correctness bench."""
    return _finite_blob(regard_eval_studies.bench_regard_eval_studies(seed))


def bench_unqover_studies_family(seed: int = _SEED + 3):
    """unqover_studies: synthetic correctness bench."""
    return _finite_blob(unqover_studies.bench_unqover_studies(seed))


def bench_vlur_studies_family(seed: int = _SEED + 4):
    """vlur_studies: synthetic correctness bench."""
    return _finite_blob(vlur_studies.bench_vlur_studies(seed))


def bench_xlsum_studies_family(seed: int = _SEED + 5):
    """xlsum_studies: synthetic correctness bench."""
    return _finite_blob(xlsum_studies.bench_xlsum_studies(seed))
