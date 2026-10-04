"""Wave-1684 bench adapters: aztec-deity-2 canon (SYNTHETIC only)."""

from quant_fund.models import (
    cihuacoatl_qa_studies,
    mayahuel_qa_studies,
    oyohualli_qa_studies,
    quetzalli_qa_studies,
    teteoinnan_qa_studies,
    yaotl_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16840


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cihuacoatl_qa_studies_family(seed: int = _SEED + 0):
    """cihuacoatl_qa_studies: synthetic correctness bench."""
    return _finite_blob(cihuacoatl_qa_studies.bench_cihuacoatl_qa_studies(seed))


def bench_mayahuel_qa_studies_family(seed: int = _SEED + 1):
    """mayahuel_qa_studies: synthetic correctness bench."""
    return _finite_blob(mayahuel_qa_studies.bench_mayahuel_qa_studies(seed))


def bench_oyohualli_qa_studies_family(seed: int = _SEED + 2):
    """oyohualli_qa_studies: synthetic correctness bench."""
    return _finite_blob(oyohualli_qa_studies.bench_oyohualli_qa_studies(seed))


def bench_quetzalli_qa_studies_family(seed: int = _SEED + 3):
    """quetzalli_qa_studies: synthetic correctness bench."""
    return _finite_blob(quetzalli_qa_studies.bench_quetzalli_qa_studies(seed))


def bench_teteoinnan_qa_studies_family(seed: int = _SEED + 4):
    """teteoinnan_qa_studies: synthetic correctness bench."""
    return _finite_blob(teteoinnan_qa_studies.bench_teteoinnan_qa_studies(seed))


def bench_yaotl_qa_studies_family(seed: int = _SEED + 5):
    """yaotl_qa_studies: synthetic correctness bench."""
    return _finite_blob(yaotl_qa_studies.bench_yaotl_qa_studies(seed))
