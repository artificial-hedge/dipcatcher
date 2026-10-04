"""Wave-1598 bench adapters: antelope-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    bontebok_qa_studies,
    bushbuck_qa_studies,
    greater_kudu_qa_studies,
    lesser_kudu_qa_studies,
    mountain_nyala_qa_studies,
    sitatunga_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 15980


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_bontebok_qa_studies_family(seed: int = _SEED + 0):
    """bontebok_qa_studies: synthetic correctness bench."""
    return _finite_blob(bontebok_qa_studies.bench_bontebok_qa_studies(seed))


def bench_bushbuck_qa_studies_family(seed: int = _SEED + 1):
    """bushbuck_qa_studies: synthetic correctness bench."""
    return _finite_blob(bushbuck_qa_studies.bench_bushbuck_qa_studies(seed))


def bench_greater_kudu_qa_studies_family(seed: int = _SEED + 2):
    """greater_kudu_qa_studies: synthetic correctness bench."""
    return _finite_blob(greater_kudu_qa_studies.bench_greater_kudu_qa_studies(seed))


def bench_lesser_kudu_qa_studies_family(seed: int = _SEED + 3):
    """lesser_kudu_qa_studies: synthetic correctness bench."""
    return _finite_blob(lesser_kudu_qa_studies.bench_lesser_kudu_qa_studies(seed))


def bench_mountain_nyala_qa_studies_family(seed: int = _SEED + 4):
    """mountain_nyala_qa_studies: synthetic correctness bench."""
    return _finite_blob(mountain_nyala_qa_studies.bench_mountain_nyala_qa_studies(seed))


def bench_sitatunga_qa_studies_family(seed: int = _SEED + 5):
    """sitatunga_qa_studies: synthetic correctness bench."""
    return _finite_blob(sitatunga_qa_studies.bench_sitatunga_qa_studies(seed))
