"""Wave-1549 bench adapters: ratite canon (SYNTHETIC only)."""

from quant_fund.models import (
    cassowary_qa_studies,
    emu_qa_studies,
    kiwi_qa_studies,
    ostrich_qa_studies,
    rhea_qa_studies,
    tinamou_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15490


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cassowary_qa_studies_family(seed: int = _SEED + 0):
    """cassowary_qa_studies: synthetic correctness bench."""
    return _finite_blob(cassowary_qa_studies.bench_cassowary_qa_studies(seed))


def bench_emu_qa_studies_family(seed: int = _SEED + 1):
    """emu_qa_studies: synthetic correctness bench."""
    return _finite_blob(emu_qa_studies.bench_emu_qa_studies(seed))


def bench_kiwi_qa_studies_family(seed: int = _SEED + 2):
    """kiwi_qa_studies: synthetic correctness bench."""
    return _finite_blob(kiwi_qa_studies.bench_kiwi_qa_studies(seed))


def bench_ostrich_qa_studies_family(seed: int = _SEED + 3):
    """ostrich_qa_studies: synthetic correctness bench."""
    return _finite_blob(ostrich_qa_studies.bench_ostrich_qa_studies(seed))


def bench_rhea_qa_studies_family(seed: int = _SEED + 4):
    """rhea_qa_studies: synthetic correctness bench."""
    return _finite_blob(rhea_qa_studies.bench_rhea_qa_studies(seed))


def bench_tinamou_qa_studies_family(seed: int = _SEED + 5):
    """tinamou_qa_studies: synthetic correctness bench."""
    return _finite_blob(tinamou_qa_studies.bench_tinamou_qa_studies(seed))
