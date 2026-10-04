"""Wave-1637 bench adapters: yokai-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    abura_sumashi_qa_studies,
    azukiarai_qa_studies,
    betobeto_2_qa_studies,
    futakuchi_qa_studies,
    rokurokubi_qa_studies,
    shirime_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 16370


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abura_sumashi_qa_studies_family(seed: int = _SEED + 0):
    """abura_sumashi_qa_studies: synthetic correctness bench."""
    return _finite_blob(abura_sumashi_qa_studies.bench_abura_sumashi_qa_studies(seed))


def bench_azukiarai_qa_studies_family(seed: int = _SEED + 1):
    """azukiarai_qa_studies: synthetic correctness bench."""
    return _finite_blob(azukiarai_qa_studies.bench_azukiarai_qa_studies(seed))


def bench_betobeto_2_qa_studies_family(seed: int = _SEED + 2):
    """betobeto_2_qa_studies: synthetic correctness bench."""
    return _finite_blob(betobeto_2_qa_studies.bench_betobeto_2_qa_studies(seed))


def bench_futakuchi_qa_studies_family(seed: int = _SEED + 3):
    """futakuchi_qa_studies: synthetic correctness bench."""
    return _finite_blob(futakuchi_qa_studies.bench_futakuchi_qa_studies(seed))


def bench_rokurokubi_qa_studies_family(seed: int = _SEED + 4):
    """rokurokubi_qa_studies: synthetic correctness bench."""
    return _finite_blob(rokurokubi_qa_studies.bench_rokurokubi_qa_studies(seed))


def bench_shirime_qa_studies_family(seed: int = _SEED + 5):
    """shirime_qa_studies: synthetic correctness bench."""
    return _finite_blob(shirime_qa_studies.bench_shirime_qa_studies(seed))
