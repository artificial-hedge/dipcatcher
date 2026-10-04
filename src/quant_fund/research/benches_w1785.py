"""Wave-1785 bench adapters: japanese-myth-6 canon (SYNTHETIC only)."""

from quant_fund.models import (
    benzaiten_qa_studies,
    hoori_qa_studies,
    jurojin_qa_studies,
    kushinadahime_qa_studies,
    toyotamahime_qa_studies,
    yamatotakeru_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17850


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_benzaiten_qa_studies_family(seed: int = _SEED + 0):
    """benzaiten_qa_studies: synthetic correctness bench."""
    return _finite_blob(benzaiten_qa_studies.bench_benzaiten_qa_studies(seed))


def bench_hoori_qa_studies_family(seed: int = _SEED + 1):
    """hoori_qa_studies: synthetic correctness bench."""
    return _finite_blob(hoori_qa_studies.bench_hoori_qa_studies(seed))


def bench_jurojin_qa_studies_family(seed: int = _SEED + 2):
    """jurojin_qa_studies: synthetic correctness bench."""
    return _finite_blob(jurojin_qa_studies.bench_jurojin_qa_studies(seed))


def bench_kushinadahime_qa_studies_family(seed: int = _SEED + 3):
    """kushinadahime_qa_studies: synthetic correctness bench."""
    return _finite_blob(kushinadahime_qa_studies.bench_kushinadahime_qa_studies(seed))


def bench_toyotamahime_qa_studies_family(seed: int = _SEED + 4):
    """toyotamahime_qa_studies: synthetic correctness bench."""
    return _finite_blob(toyotamahime_qa_studies.bench_toyotamahime_qa_studies(seed))


def bench_yamatotakeru_qa_studies_family(seed: int = _SEED + 5):
    """yamatotakeru_qa_studies: synthetic correctness bench."""
    return _finite_blob(yamatotakeru_qa_studies.bench_yamatotakeru_qa_studies(seed))
