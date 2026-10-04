"""Wave-1893 bench adapters: mesopotamian-demon canon (SYNTHETIC only)."""

from quant_fund.models import (
    alu_demon_qa_studies,
    ardat_lili_qa_studies,
    galla_demon_qa_studies,
    lamashtu_qa_studies,
    lilitu_qa_studies,
    rabisu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18930


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_alu_demon_qa_studies_family(seed: int = _SEED + 0):
    """alu_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(alu_demon_qa_studies.bench_alu_demon_qa_studies(seed))


def bench_ardat_lili_qa_studies_family(seed: int = _SEED + 1):
    """ardat_lili_qa_studies: synthetic correctness bench."""
    return _finite_blob(ardat_lili_qa_studies.bench_ardat_lili_qa_studies(seed))


def bench_galla_demon_qa_studies_family(seed: int = _SEED + 2):
    """galla_demon_qa_studies: synthetic correctness bench."""
    return _finite_blob(galla_demon_qa_studies.bench_galla_demon_qa_studies(seed))


def bench_lamashtu_qa_studies_family(seed: int = _SEED + 3):
    """lamashtu_qa_studies: synthetic correctness bench."""
    return _finite_blob(lamashtu_qa_studies.bench_lamashtu_qa_studies(seed))


def bench_lilitu_qa_studies_family(seed: int = _SEED + 4):
    """lilitu_qa_studies: synthetic correctness bench."""
    return _finite_blob(lilitu_qa_studies.bench_lilitu_qa_studies(seed))


def bench_rabisu_qa_studies_family(seed: int = _SEED + 5):
    """rabisu_qa_studies: synthetic correctness bench."""
    return _finite_blob(rabisu_qa_studies.bench_rabisu_qa_studies(seed))
