"""Wave-1812 bench adapters: aztec-deity-5 canon (SYNTHETIC only)."""

from quant_fund.models import (
    centeotl2_qa_studies,
    mayahuel2_qa_studies,
    mixcoatl2_qa_studies,
    tlaloc2_qa_studies,
    xipe2_qa_studies,
    xochipilli2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18120


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_centeotl2_qa_studies_family(seed: int = _SEED + 0):
    """centeotl2_qa_studies: synthetic correctness bench."""
    return _finite_blob(centeotl2_qa_studies.bench_centeotl2_qa_studies(seed))


def bench_mayahuel2_qa_studies_family(seed: int = _SEED + 1):
    """mayahuel2_qa_studies: synthetic correctness bench."""
    return _finite_blob(mayahuel2_qa_studies.bench_mayahuel2_qa_studies(seed))


def bench_mixcoatl2_qa_studies_family(seed: int = _SEED + 2):
    """mixcoatl2_qa_studies: synthetic correctness bench."""
    return _finite_blob(mixcoatl2_qa_studies.bench_mixcoatl2_qa_studies(seed))


def bench_tlaloc2_qa_studies_family(seed: int = _SEED + 3):
    """tlaloc2_qa_studies: synthetic correctness bench."""
    return _finite_blob(tlaloc2_qa_studies.bench_tlaloc2_qa_studies(seed))


def bench_xipe2_qa_studies_family(seed: int = _SEED + 4):
    """xipe2_qa_studies: synthetic correctness bench."""
    return _finite_blob(xipe2_qa_studies.bench_xipe2_qa_studies(seed))


def bench_xochipilli2_qa_studies_family(seed: int = _SEED + 5):
    """xochipilli2_qa_studies: synthetic correctness bench."""
    return _finite_blob(xochipilli2_qa_studies.bench_xochipilli2_qa_studies(seed))
