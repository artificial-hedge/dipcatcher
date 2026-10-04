"""Wave-1732 bench adapters: polish-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    dziewanna_qa_studies,
    marzanna_qa_studies,
    mokosz_qa_studies,
    nija_qa_studies,
    swarozyc_qa_studies,
    zywie_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17320


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dziewanna_qa_studies_family(seed: int = _SEED + 0):
    """dziewanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(dziewanna_qa_studies.bench_dziewanna_qa_studies(seed))


def bench_marzanna_qa_studies_family(seed: int = _SEED + 1):
    """marzanna_qa_studies: synthetic correctness bench."""
    return _finite_blob(marzanna_qa_studies.bench_marzanna_qa_studies(seed))


def bench_mokosz_qa_studies_family(seed: int = _SEED + 2):
    """mokosz_qa_studies: synthetic correctness bench."""
    return _finite_blob(mokosz_qa_studies.bench_mokosz_qa_studies(seed))


def bench_nija_qa_studies_family(seed: int = _SEED + 3):
    """nija_qa_studies: synthetic correctness bench."""
    return _finite_blob(nija_qa_studies.bench_nija_qa_studies(seed))


def bench_swarozyc_qa_studies_family(seed: int = _SEED + 4):
    """swarozyc_qa_studies: synthetic correctness bench."""
    return _finite_blob(swarozyc_qa_studies.bench_swarozyc_qa_studies(seed))


def bench_zywie_qa_studies_family(seed: int = _SEED + 5):
    """zywie_qa_studies: synthetic correctness bench."""
    return _finite_blob(zywie_qa_studies.bench_zywie_qa_studies(seed))
