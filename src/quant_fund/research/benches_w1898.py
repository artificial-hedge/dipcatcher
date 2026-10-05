"""Wave-1898 bench adapters: malay-archipelago-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    bajang_qa_studies,
    kum_kum_qa_studies,
    pelesit_qa_studies,
    penanggalan_qa_studies,
    pontianak_qa_studies,
    toyol_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18980


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bajang_qa_studies_family(seed: int = _SEED + 0):
    """bajang_qa_studies: synthetic correctness bench."""
    return _finite_blob(bajang_qa_studies.bench_bajang_qa_studies(seed))


def bench_kum_kum_qa_studies_family(seed: int = _SEED + 1):
    """kum_kum_qa_studies: synthetic correctness bench."""
    return _finite_blob(kum_kum_qa_studies.bench_kum_kum_qa_studies(seed))


def bench_pelesit_qa_studies_family(seed: int = _SEED + 2):
    """pelesit_qa_studies: synthetic correctness bench."""
    return _finite_blob(pelesit_qa_studies.bench_pelesit_qa_studies(seed))


def bench_penanggalan_qa_studies_family(seed: int = _SEED + 3):
    """penanggalan_qa_studies: synthetic correctness bench."""
    return _finite_blob(penanggalan_qa_studies.bench_penanggalan_qa_studies(seed))


def bench_pontianak_qa_studies_family(seed: int = _SEED + 4):
    """pontianak_qa_studies: synthetic correctness bench."""
    return _finite_blob(pontianak_qa_studies.bench_pontianak_qa_studies(seed))


def bench_toyol_qa_studies_family(seed: int = _SEED + 5):
    """toyol_qa_studies: synthetic correctness bench."""
    return _finite_blob(toyol_qa_studies.bench_toyol_qa_studies(seed))
