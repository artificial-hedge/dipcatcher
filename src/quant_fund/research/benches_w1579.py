"""Wave-1579 bench adapters: dwarf-antelope canon (SYNTHETIC only)."""

from quant_fund.models import (
    dikdik_qa_studies,
    grysbok_qa_studies,
    klipspringer_qa_studies,
    rhebok_qa_studies,
    steenbok_qa_studies,
    suni_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15790


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dikdik_qa_studies_family(seed: int = _SEED + 0):
    """dikdik_qa_studies: synthetic correctness bench."""
    return _finite_blob(dikdik_qa_studies.bench_dikdik_qa_studies(seed))


def bench_grysbok_qa_studies_family(seed: int = _SEED + 1):
    """grysbok_qa_studies: synthetic correctness bench."""
    return _finite_blob(grysbok_qa_studies.bench_grysbok_qa_studies(seed))


def bench_klipspringer_qa_studies_family(seed: int = _SEED + 2):
    """klipspringer_qa_studies: synthetic correctness bench."""
    return _finite_blob(klipspringer_qa_studies.bench_klipspringer_qa_studies(seed))


def bench_rhebok_qa_studies_family(seed: int = _SEED + 3):
    """rhebok_qa_studies: synthetic correctness bench."""
    return _finite_blob(rhebok_qa_studies.bench_rhebok_qa_studies(seed))


def bench_steenbok_qa_studies_family(seed: int = _SEED + 4):
    """steenbok_qa_studies: synthetic correctness bench."""
    return _finite_blob(steenbok_qa_studies.bench_steenbok_qa_studies(seed))


def bench_suni_qa_studies_family(seed: int = _SEED + 5):
    """suni_qa_studies: synthetic correctness bench."""
    return _finite_blob(suni_qa_studies.bench_suni_qa_studies(seed))
