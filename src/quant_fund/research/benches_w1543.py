"""Wave-1543 bench adapters: nightjar-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    oilbird_qa_studies,
    owlet_nightjar_qa_studies,
    pauraque_qa_studies,
    poorwill_qa_studies,
    potoo_qa_studies,
    whip_poor_will_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15430


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_oilbird_qa_studies_family(seed: int = _SEED + 0):
    """oilbird_qa_studies: synthetic correctness bench."""
    return _finite_blob(oilbird_qa_studies.bench_oilbird_qa_studies(seed))


def bench_owlet_nightjar_qa_studies_family(seed: int = _SEED + 1):
    """owlet_nightjar_qa_studies: synthetic correctness bench."""
    return _finite_blob(owlet_nightjar_qa_studies.bench_owlet_nightjar_qa_studies(seed))


def bench_pauraque_qa_studies_family(seed: int = _SEED + 2):
    """pauraque_qa_studies: synthetic correctness bench."""
    return _finite_blob(pauraque_qa_studies.bench_pauraque_qa_studies(seed))


def bench_poorwill_qa_studies_family(seed: int = _SEED + 3):
    """poorwill_qa_studies: synthetic correctness bench."""
    return _finite_blob(poorwill_qa_studies.bench_poorwill_qa_studies(seed))


def bench_potoo_qa_studies_family(seed: int = _SEED + 4):
    """potoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(potoo_qa_studies.bench_potoo_qa_studies(seed))


def bench_whip_poor_will_qa_studies_family(seed: int = _SEED + 5):
    """whip_poor_will_qa_studies: synthetic correctness bench."""
    return _finite_blob(whip_poor_will_qa_studies.bench_whip_poor_will_qa_studies(seed))
