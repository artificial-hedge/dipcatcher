"""Wave-1817 bench adapters: hindu-myth-7 canon (SYNTHETIC only)."""

from quant_fund.models import (
    durga2_qa_studies,
    ganga2_qa_studies,
    kali2_qa_studies,
    lakshmi2_qa_studies,
    parvati2_qa_studies,
    saraswati2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18170


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_durga2_qa_studies_family(seed: int = _SEED + 0):
    """durga2_qa_studies: synthetic correctness bench."""
    return _finite_blob(durga2_qa_studies.bench_durga2_qa_studies(seed))


def bench_ganga2_qa_studies_family(seed: int = _SEED + 1):
    """ganga2_qa_studies: synthetic correctness bench."""
    return _finite_blob(ganga2_qa_studies.bench_ganga2_qa_studies(seed))


def bench_kali2_qa_studies_family(seed: int = _SEED + 2):
    """kali2_qa_studies: synthetic correctness bench."""
    return _finite_blob(kali2_qa_studies.bench_kali2_qa_studies(seed))


def bench_lakshmi2_qa_studies_family(seed: int = _SEED + 3):
    """lakshmi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(lakshmi2_qa_studies.bench_lakshmi2_qa_studies(seed))


def bench_parvati2_qa_studies_family(seed: int = _SEED + 4):
    """parvati2_qa_studies: synthetic correctness bench."""
    return _finite_blob(parvati2_qa_studies.bench_parvati2_qa_studies(seed))


def bench_saraswati2_qa_studies_family(seed: int = _SEED + 5):
    """saraswati2_qa_studies: synthetic correctness bench."""
    return _finite_blob(saraswati2_qa_studies.bench_saraswati2_qa_studies(seed))
