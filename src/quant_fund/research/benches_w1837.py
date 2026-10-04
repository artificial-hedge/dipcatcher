"""Wave-1837 bench adapters: lycian-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    apollopatara2_qa_studies,
    eliyana2_qa_studies,
    erbbina2_qa_studies,
    leto2_qa_studies,
    trqqiz2_qa_studies,
    xssentimi2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18370


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_apollopatara2_qa_studies_family(seed: int = _SEED + 0):
    """apollopatara2_qa_studies: synthetic correctness bench."""
    return _finite_blob(apollopatara2_qa_studies.bench_apollopatara2_qa_studies(seed))


def bench_eliyana2_qa_studies_family(seed: int = _SEED + 1):
    """eliyana2_qa_studies: synthetic correctness bench."""
    return _finite_blob(eliyana2_qa_studies.bench_eliyana2_qa_studies(seed))


def bench_erbbina2_qa_studies_family(seed: int = _SEED + 2):
    """erbbina2_qa_studies: synthetic correctness bench."""
    return _finite_blob(erbbina2_qa_studies.bench_erbbina2_qa_studies(seed))


def bench_leto2_qa_studies_family(seed: int = _SEED + 3):
    """leto2_qa_studies: synthetic correctness bench."""
    return _finite_blob(leto2_qa_studies.bench_leto2_qa_studies(seed))


def bench_trqqiz2_qa_studies_family(seed: int = _SEED + 4):
    """trqqiz2_qa_studies: synthetic correctness bench."""
    return _finite_blob(trqqiz2_qa_studies.bench_trqqiz2_qa_studies(seed))


def bench_xssentimi2_qa_studies_family(seed: int = _SEED + 5):
    """xssentimi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(xssentimi2_qa_studies.bench_xssentimi2_qa_studies(seed))
