"""Wave-1473 bench adapters: spice canon (SYNTHETIC only)."""

from quant_fund.models import (
    clove_qa_studies,
    dill_qa_studies,
    fennel_qa_studies,
    lemongrass_qa_studies,
    mint_qa_studies,
    nutmeg_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14730


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_clove_qa_studies_family(seed: int = _SEED + 0):
    """clove_qa_studies: synthetic correctness bench."""
    return _finite_blob(clove_qa_studies.bench_clove_qa_studies(seed))


def bench_dill_qa_studies_family(seed: int = _SEED + 1):
    """dill_qa_studies: synthetic correctness bench."""
    return _finite_blob(dill_qa_studies.bench_dill_qa_studies(seed))


def bench_fennel_qa_studies_family(seed: int = _SEED + 2):
    """fennel_qa_studies: synthetic correctness bench."""
    return _finite_blob(fennel_qa_studies.bench_fennel_qa_studies(seed))


def bench_lemongrass_qa_studies_family(seed: int = _SEED + 3):
    """lemongrass_qa_studies: synthetic correctness bench."""
    return _finite_blob(lemongrass_qa_studies.bench_lemongrass_qa_studies(seed))


def bench_mint_qa_studies_family(seed: int = _SEED + 4):
    """mint_qa_studies: synthetic correctness bench."""
    return _finite_blob(mint_qa_studies.bench_mint_qa_studies(seed))


def bench_nutmeg_qa_studies_family(seed: int = _SEED + 5):
    """nutmeg_qa_studies: synthetic correctness bench."""
    return _finite_blob(nutmeg_qa_studies.bench_nutmeg_qa_studies(seed))
