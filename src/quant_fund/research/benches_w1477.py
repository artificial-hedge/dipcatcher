"""Wave-1477 bench adapters: seabird canon (SYNTHETIC only)."""

from quant_fund.models import (
    albatross_qa_studies,
    gannet_qa_studies,
    petrel_qa_studies,
    puffin_qa_studies,
    shearwater_qa_studies,
    skua_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14770


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_albatross_qa_studies_family(seed: int = _SEED + 0):
    """albatross_qa_studies: synthetic correctness bench."""
    return _finite_blob(albatross_qa_studies.bench_albatross_qa_studies(seed))


def bench_gannet_qa_studies_family(seed: int = _SEED + 1):
    """gannet_qa_studies: synthetic correctness bench."""
    return _finite_blob(gannet_qa_studies.bench_gannet_qa_studies(seed))


def bench_petrel_qa_studies_family(seed: int = _SEED + 2):
    """petrel_qa_studies: synthetic correctness bench."""
    return _finite_blob(petrel_qa_studies.bench_petrel_qa_studies(seed))


def bench_puffin_qa_studies_family(seed: int = _SEED + 3):
    """puffin_qa_studies: synthetic correctness bench."""
    return _finite_blob(puffin_qa_studies.bench_puffin_qa_studies(seed))


def bench_shearwater_qa_studies_family(seed: int = _SEED + 4):
    """shearwater_qa_studies: synthetic correctness bench."""
    return _finite_blob(shearwater_qa_studies.bench_shearwater_qa_studies(seed))


def bench_skua_qa_studies_family(seed: int = _SEED + 5):
    """skua_qa_studies: synthetic correctness bench."""
    return _finite_blob(skua_qa_studies.bench_skua_qa_studies(seed))
