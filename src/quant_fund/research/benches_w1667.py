"""Wave-1667 bench adapters: greek-nature canon (SYNTHETIC only)."""

from quant_fund.models import (
    centauride_qa_studies,
    dryad_qa_studies,
    faun_qa_studies,
    hamadryad_qa_studies,
    nereid_qa_studies,
    nymph_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16670


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_centauride_qa_studies_family(seed: int = _SEED + 0):
    """centauride_qa_studies: synthetic correctness bench."""
    return _finite_blob(centauride_qa_studies.bench_centauride_qa_studies(seed))


def bench_dryad_qa_studies_family(seed: int = _SEED + 1):
    """dryad_qa_studies: synthetic correctness bench."""
    return _finite_blob(dryad_qa_studies.bench_dryad_qa_studies(seed))


def bench_faun_qa_studies_family(seed: int = _SEED + 2):
    """faun_qa_studies: synthetic correctness bench."""
    return _finite_blob(faun_qa_studies.bench_faun_qa_studies(seed))


def bench_hamadryad_qa_studies_family(seed: int = _SEED + 3):
    """hamadryad_qa_studies: synthetic correctness bench."""
    return _finite_blob(hamadryad_qa_studies.bench_hamadryad_qa_studies(seed))


def bench_nereid_qa_studies_family(seed: int = _SEED + 4):
    """nereid_qa_studies: synthetic correctness bench."""
    return _finite_blob(nereid_qa_studies.bench_nereid_qa_studies(seed))


def bench_nymph_qa_studies_family(seed: int = _SEED + 5):
    """nymph_qa_studies: synthetic correctness bench."""
    return _finite_blob(nymph_qa_studies.bench_nymph_qa_studies(seed))
