"""Wave-1576 bench adapters: mammal canon (SYNTHETIC only)."""

from quant_fund.models import (
    binturong_qa_studies,
    fossa_qa_studies,
    honey_badger_qa_studies,
    kusimanse_qa_studies,
    maned_wolf_qa_studies,
    sun_bear_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15760


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_binturong_qa_studies_family(seed: int = _SEED + 0):
    """binturong_qa_studies: synthetic correctness bench."""
    return _finite_blob(binturong_qa_studies.bench_binturong_qa_studies(seed))


def bench_fossa_qa_studies_family(seed: int = _SEED + 1):
    """fossa_qa_studies: synthetic correctness bench."""
    return _finite_blob(fossa_qa_studies.bench_fossa_qa_studies(seed))


def bench_honey_badger_qa_studies_family(seed: int = _SEED + 2):
    """honey_badger_qa_studies: synthetic correctness bench."""
    return _finite_blob(honey_badger_qa_studies.bench_honey_badger_qa_studies(seed))


def bench_kusimanse_qa_studies_family(seed: int = _SEED + 3):
    """kusimanse_qa_studies: synthetic correctness bench."""
    return _finite_blob(kusimanse_qa_studies.bench_kusimanse_qa_studies(seed))


def bench_maned_wolf_qa_studies_family(seed: int = _SEED + 4):
    """maned_wolf_qa_studies: synthetic correctness bench."""
    return _finite_blob(maned_wolf_qa_studies.bench_maned_wolf_qa_studies(seed))


def bench_sun_bear_qa_studies_family(seed: int = _SEED + 5):
    """sun_bear_qa_studies: synthetic correctness bench."""
    return _finite_blob(sun_bear_qa_studies.bench_sun_bear_qa_studies(seed))
