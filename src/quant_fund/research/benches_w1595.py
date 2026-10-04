"""Wave-1595 bench adapters: felid-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    andean_cat_qa_studies,
    bay_cat_qa_studies,
    flat_headed_qa_studies,
    geoffroys_qa_studies,
    marbled_cat_qa_studies,
    pampas_cat_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15950


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_andean_cat_qa_studies_family(seed: int = _SEED + 0):
    """andean_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(andean_cat_qa_studies.bench_andean_cat_qa_studies(seed))


def bench_bay_cat_qa_studies_family(seed: int = _SEED + 1):
    """bay_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(bay_cat_qa_studies.bench_bay_cat_qa_studies(seed))


def bench_flat_headed_qa_studies_family(seed: int = _SEED + 2):
    """flat_headed_qa_studies: synthetic correctness bench."""
    return _finite_blob(flat_headed_qa_studies.bench_flat_headed_qa_studies(seed))


def bench_geoffroys_qa_studies_family(seed: int = _SEED + 3):
    """geoffroys_qa_studies: synthetic correctness bench."""
    return _finite_blob(geoffroys_qa_studies.bench_geoffroys_qa_studies(seed))


def bench_marbled_cat_qa_studies_family(seed: int = _SEED + 4):
    """marbled_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(marbled_cat_qa_studies.bench_marbled_cat_qa_studies(seed))


def bench_pampas_cat_qa_studies_family(seed: int = _SEED + 5):
    """pampas_cat_qa_studies: synthetic correctness bench."""
    return _finite_blob(pampas_cat_qa_studies.bench_pampas_cat_qa_studies(seed))
