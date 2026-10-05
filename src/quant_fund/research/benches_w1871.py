"""Wave-1871 bench adapters: manx-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    buggane_qa_studies,
    fenodyree_qa_studies,
    glashtyn_qa_studies,
    moddey_dhoo_qa_studies,
    phynnodderee_qa_studies,
    tarroo_ushtey_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18710


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_buggane_qa_studies_family(seed: int = _SEED + 0):
    """buggane_qa_studies: synthetic correctness bench."""
    return _finite_blob(buggane_qa_studies.bench_buggane_qa_studies(seed))


def bench_fenodyree_qa_studies_family(seed: int = _SEED + 1):
    """fenodyree_qa_studies: synthetic correctness bench."""
    return _finite_blob(fenodyree_qa_studies.bench_fenodyree_qa_studies(seed))


def bench_glashtyn_qa_studies_family(seed: int = _SEED + 2):
    """glashtyn_qa_studies: synthetic correctness bench."""
    return _finite_blob(glashtyn_qa_studies.bench_glashtyn_qa_studies(seed))


def bench_moddey_dhoo_qa_studies_family(seed: int = _SEED + 3):
    """moddey_dhoo_qa_studies: synthetic correctness bench."""
    return _finite_blob(moddey_dhoo_qa_studies.bench_moddey_dhoo_qa_studies(seed))


def bench_phynnodderee_qa_studies_family(seed: int = _SEED + 4):
    """phynnodderee_qa_studies: synthetic correctness bench."""
    return _finite_blob(phynnodderee_qa_studies.bench_phynnodderee_qa_studies(seed))


def bench_tarroo_ushtey_qa_studies_family(seed: int = _SEED + 5):
    """tarroo_ushtey_qa_studies: synthetic correctness bench."""
    return _finite_blob(tarroo_ushtey_qa_studies.bench_tarroo_ushtey_qa_studies(seed))
