"""Wave-1624 bench adapters: tundra canon (SYNTHETIC only)."""

from quant_fund.models import (
    arctic_hare_qa_studies,
    gyrfalcon_qa_studies,
    pallas_manul_qa_studies,
    ptarmigan_qa_studies,
    snowshoe_qa_studies,
    tundra_swan_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16240


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_arctic_hare_qa_studies_family(seed: int = _SEED + 0):
    """arctic_hare_qa_studies: synthetic correctness bench."""
    return _finite_blob(arctic_hare_qa_studies.bench_arctic_hare_qa_studies(seed))


def bench_gyrfalcon_qa_studies_family(seed: int = _SEED + 1):
    """gyrfalcon_qa_studies: synthetic correctness bench."""
    return _finite_blob(gyrfalcon_qa_studies.bench_gyrfalcon_qa_studies(seed))


def bench_pallas_manul_qa_studies_family(seed: int = _SEED + 2):
    """pallas_manul_qa_studies: synthetic correctness bench."""
    return _finite_blob(pallas_manul_qa_studies.bench_pallas_manul_qa_studies(seed))


def bench_ptarmigan_qa_studies_family(seed: int = _SEED + 3):
    """ptarmigan_qa_studies: synthetic correctness bench."""
    return _finite_blob(ptarmigan_qa_studies.bench_ptarmigan_qa_studies(seed))


def bench_snowshoe_qa_studies_family(seed: int = _SEED + 4):
    """snowshoe_qa_studies: synthetic correctness bench."""
    return _finite_blob(snowshoe_qa_studies.bench_snowshoe_qa_studies(seed))


def bench_tundra_swan_qa_studies_family(seed: int = _SEED + 5):
    """tundra_swan_qa_studies: synthetic correctness bench."""
    return _finite_blob(tundra_swan_qa_studies.bench_tundra_swan_qa_studies(seed))
