"""Wave-1829 bench adapters: nenets-myth-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    iljang2_qa_studies,
    nemlert2_qa_studies,
    numgum2_qa_studies,
    otysi2_qa_studies,
    parnae2_qa_studies,
    xiberi2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18290


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_iljang2_qa_studies_family(seed: int = _SEED + 0):
    """iljang2_qa_studies: synthetic correctness bench."""
    return _finite_blob(iljang2_qa_studies.bench_iljang2_qa_studies(seed))


def bench_nemlert2_qa_studies_family(seed: int = _SEED + 1):
    """nemlert2_qa_studies: synthetic correctness bench."""
    return _finite_blob(nemlert2_qa_studies.bench_nemlert2_qa_studies(seed))


def bench_numgum2_qa_studies_family(seed: int = _SEED + 2):
    """numgum2_qa_studies: synthetic correctness bench."""
    return _finite_blob(numgum2_qa_studies.bench_numgum2_qa_studies(seed))


def bench_otysi2_qa_studies_family(seed: int = _SEED + 3):
    """otysi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(otysi2_qa_studies.bench_otysi2_qa_studies(seed))


def bench_parnae2_qa_studies_family(seed: int = _SEED + 4):
    """parnae2_qa_studies: synthetic correctness bench."""
    return _finite_blob(parnae2_qa_studies.bench_parnae2_qa_studies(seed))


def bench_xiberi2_qa_studies_family(seed: int = _SEED + 5):
    """xiberi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(xiberi2_qa_studies.bench_xiberi2_qa_studies(seed))
