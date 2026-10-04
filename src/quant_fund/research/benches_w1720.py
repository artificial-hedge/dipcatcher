"""Wave-1720 bench adapters: sami-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    akka_qa_studies,
    juksakka_qa_studies,
    lieaibolmmai_qa_studies,
    radien_qa_studies,
    sarakka_qa_studies,
    ukso_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17200


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_akka_qa_studies_family(seed: int = _SEED + 0):
    """akka_qa_studies: synthetic correctness bench."""
    return _finite_blob(akka_qa_studies.bench_akka_qa_studies(seed))


def bench_juksakka_qa_studies_family(seed: int = _SEED + 1):
    """juksakka_qa_studies: synthetic correctness bench."""
    return _finite_blob(juksakka_qa_studies.bench_juksakka_qa_studies(seed))


def bench_lieaibolmmai_qa_studies_family(seed: int = _SEED + 2):
    """lieaibolmmai_qa_studies: synthetic correctness bench."""
    return _finite_blob(lieaibolmmai_qa_studies.bench_lieaibolmmai_qa_studies(seed))


def bench_radien_qa_studies_family(seed: int = _SEED + 3):
    """radien_qa_studies: synthetic correctness bench."""
    return _finite_blob(radien_qa_studies.bench_radien_qa_studies(seed))


def bench_sarakka_qa_studies_family(seed: int = _SEED + 4):
    """sarakka_qa_studies: synthetic correctness bench."""
    return _finite_blob(sarakka_qa_studies.bench_sarakka_qa_studies(seed))


def bench_ukso_qa_studies_family(seed: int = _SEED + 5):
    """ukso_qa_studies: synthetic correctness bench."""
    return _finite_blob(ukso_qa_studies.bench_ukso_qa_studies(seed))
