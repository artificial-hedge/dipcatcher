"""Wave-1844 bench adapters: aramaean-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    aram2_qa_studies,
    ashima_qa_studies,
    baalshamin_qa_studies,
    resheph2_qa_studies,
    rimmon_qa_studies,
    sahr_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18440


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_aram2_qa_studies_family(seed: int = _SEED + 0):
    """aram2_qa_studies: synthetic correctness bench."""
    return _finite_blob(aram2_qa_studies.bench_aram2_qa_studies(seed))


def bench_ashima_qa_studies_family(seed: int = _SEED + 1):
    """ashima_qa_studies: synthetic correctness bench."""
    return _finite_blob(ashima_qa_studies.bench_ashima_qa_studies(seed))


def bench_baalshamin_qa_studies_family(seed: int = _SEED + 2):
    """baalshamin_qa_studies: synthetic correctness bench."""
    return _finite_blob(baalshamin_qa_studies.bench_baalshamin_qa_studies(seed))


def bench_resheph2_qa_studies_family(seed: int = _SEED + 3):
    """resheph2_qa_studies: synthetic correctness bench."""
    return _finite_blob(resheph2_qa_studies.bench_resheph2_qa_studies(seed))


def bench_rimmon_qa_studies_family(seed: int = _SEED + 4):
    """rimmon_qa_studies: synthetic correctness bench."""
    return _finite_blob(rimmon_qa_studies.bench_rimmon_qa_studies(seed))


def bench_sahr_qa_studies_family(seed: int = _SEED + 5):
    """sahr_qa_studies: synthetic correctness bench."""
    return _finite_blob(sahr_qa_studies.bench_sahr_qa_studies(seed))
