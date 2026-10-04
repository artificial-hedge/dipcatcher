"""Wave-1659 bench adapters: philippine-beast canon (SYNTHETIC only)."""

from quant_fund.models import (
    manananggal_qa_studies,
    minokawa_qa_studies,
    nuno_qa_studies,
    siyokoy_qa_studies,
    tiyanak_qa_studies,
    wakwak_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16590


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_manananggal_qa_studies_family(seed: int = _SEED + 0):
    """manananggal_qa_studies: synthetic correctness bench."""
    return _finite_blob(manananggal_qa_studies.bench_manananggal_qa_studies(seed))


def bench_minokawa_qa_studies_family(seed: int = _SEED + 1):
    """minokawa_qa_studies: synthetic correctness bench."""
    return _finite_blob(minokawa_qa_studies.bench_minokawa_qa_studies(seed))


def bench_nuno_qa_studies_family(seed: int = _SEED + 2):
    """nuno_qa_studies: synthetic correctness bench."""
    return _finite_blob(nuno_qa_studies.bench_nuno_qa_studies(seed))


def bench_siyokoy_qa_studies_family(seed: int = _SEED + 3):
    """siyokoy_qa_studies: synthetic correctness bench."""
    return _finite_blob(siyokoy_qa_studies.bench_siyokoy_qa_studies(seed))


def bench_tiyanak_qa_studies_family(seed: int = _SEED + 4):
    """tiyanak_qa_studies: synthetic correctness bench."""
    return _finite_blob(tiyanak_qa_studies.bench_tiyanak_qa_studies(seed))


def bench_wakwak_qa_studies_family(seed: int = _SEED + 5):
    """wakwak_qa_studies: synthetic correctness bench."""
    return _finite_blob(wakwak_qa_studies.bench_wakwak_qa_studies(seed))
