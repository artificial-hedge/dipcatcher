"""Wave-1944 bench adapters: korean-gwishin canon (SYNTHETIC only)."""

from quant_fund.models import (
    cheonyeo_gwishin_qa_studies,
    dokkaebi_qa_studies,
    gumiho_qa_studies,
    gwishin_qa_studies,
    mul_gwishin_qa_studies,
    oeggwi_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 19440


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_cheonyeo_gwishin_qa_studies_family(seed: int = _SEED + 0):
    """cheonyeo_gwishin_qa_studies: synthetic correctness bench."""
    return _finite_blob(cheonyeo_gwishin_qa_studies.bench_cheonyeo_gwishin_qa_studies(seed))


def bench_dokkaebi_qa_studies_family(seed: int = _SEED + 1):
    """dokkaebi_qa_studies: synthetic correctness bench."""
    return _finite_blob(dokkaebi_qa_studies.bench_dokkaebi_qa_studies(seed))


def bench_gumiho_qa_studies_family(seed: int = _SEED + 2):
    """gumiho_qa_studies: synthetic correctness bench."""
    return _finite_blob(gumiho_qa_studies.bench_gumiho_qa_studies(seed))


def bench_gwishin_qa_studies_family(seed: int = _SEED + 3):
    """gwishin_qa_studies: synthetic correctness bench."""
    return _finite_blob(gwishin_qa_studies.bench_gwishin_qa_studies(seed))


def bench_mul_gwishin_qa_studies_family(seed: int = _SEED + 4):
    """mul_gwishin_qa_studies: synthetic correctness bench."""
    return _finite_blob(mul_gwishin_qa_studies.bench_mul_gwishin_qa_studies(seed))


def bench_oeggwi_qa_studies_family(seed: int = _SEED + 5):
    """oeggwi_qa_studies: synthetic correctness bench."""
    return _finite_blob(oeggwi_qa_studies.bench_oeggwi_qa_studies(seed))
