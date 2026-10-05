"""Wave-1921 bench adapters: baltic-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    baubas_qa_studies,
    kaukas_qa_studies,
    lauma_qa_studies,
    pukis_qa_studies,
    spigana_qa_studies,
    vilkacis_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19210


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baubas_qa_studies_family(seed: int = _SEED + 0):
    """baubas_qa_studies: synthetic correctness bench."""
    return _finite_blob(baubas_qa_studies.bench_baubas_qa_studies(seed))


def bench_kaukas_qa_studies_family(seed: int = _SEED + 1):
    """kaukas_qa_studies: synthetic correctness bench."""
    return _finite_blob(kaukas_qa_studies.bench_kaukas_qa_studies(seed))


def bench_lauma_qa_studies_family(seed: int = _SEED + 2):
    """lauma_qa_studies: synthetic correctness bench."""
    return _finite_blob(lauma_qa_studies.bench_lauma_qa_studies(seed))


def bench_pukis_qa_studies_family(seed: int = _SEED + 3):
    """pukis_qa_studies: synthetic correctness bench."""
    return _finite_blob(pukis_qa_studies.bench_pukis_qa_studies(seed))


def bench_spigana_qa_studies_family(seed: int = _SEED + 4):
    """spigana_qa_studies: synthetic correctness bench."""
    return _finite_blob(spigana_qa_studies.bench_spigana_qa_studies(seed))


def bench_vilkacis_qa_studies_family(seed: int = _SEED + 5):
    """vilkacis_qa_studies: synthetic correctness bench."""
    return _finite_blob(vilkacis_qa_studies.bench_vilkacis_qa_studies(seed))
