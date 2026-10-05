"""Wave-1855 bench adapters: iberian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    ataecina_qa_studies,
    bandua_qa_studies,
    cariocecus_qa_studies,
    endovellicus_qa_studies,
    nabia_qa_studies,
    trebaruna_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18550


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_ataecina_qa_studies_family(seed: int = _SEED + 0):
    """ataecina_qa_studies: synthetic correctness bench."""
    return _finite_blob(ataecina_qa_studies.bench_ataecina_qa_studies(seed))


def bench_bandua_qa_studies_family(seed: int = _SEED + 1):
    """bandua_qa_studies: synthetic correctness bench."""
    return _finite_blob(bandua_qa_studies.bench_bandua_qa_studies(seed))


def bench_cariocecus_qa_studies_family(seed: int = _SEED + 2):
    """cariocecus_qa_studies: synthetic correctness bench."""
    return _finite_blob(cariocecus_qa_studies.bench_cariocecus_qa_studies(seed))


def bench_endovellicus_qa_studies_family(seed: int = _SEED + 3):
    """endovellicus_qa_studies: synthetic correctness bench."""
    return _finite_blob(endovellicus_qa_studies.bench_endovellicus_qa_studies(seed))


def bench_nabia_qa_studies_family(seed: int = _SEED + 4):
    """nabia_qa_studies: synthetic correctness bench."""
    return _finite_blob(nabia_qa_studies.bench_nabia_qa_studies(seed))


def bench_trebaruna_qa_studies_family(seed: int = _SEED + 5):
    """trebaruna_qa_studies: synthetic correctness bench."""
    return _finite_blob(trebaruna_qa_studies.bench_trebaruna_qa_studies(seed))
