"""Wave-1822 bench adapters: korean-myth-3 canon (SYNTHETIC only)."""

from quant_fund.models import (
    baridegi2_qa_studies,
    dangun2_qa_studies,
    dolhareubang2_qa_studies,
    gamunjang2_qa_studies,
    hwanung2_qa_studies,
    jacheongbi2_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 18220


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_baridegi2_qa_studies_family(seed: int = _SEED + 0):
    """baridegi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(baridegi2_qa_studies.bench_baridegi2_qa_studies(seed))


def bench_dangun2_qa_studies_family(seed: int = _SEED + 1):
    """dangun2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dangun2_qa_studies.bench_dangun2_qa_studies(seed))


def bench_dolhareubang2_qa_studies_family(seed: int = _SEED + 2):
    """dolhareubang2_qa_studies: synthetic correctness bench."""
    return _finite_blob(dolhareubang2_qa_studies.bench_dolhareubang2_qa_studies(seed))


def bench_gamunjang2_qa_studies_family(seed: int = _SEED + 3):
    """gamunjang2_qa_studies: synthetic correctness bench."""
    return _finite_blob(gamunjang2_qa_studies.bench_gamunjang2_qa_studies(seed))


def bench_hwanung2_qa_studies_family(seed: int = _SEED + 4):
    """hwanung2_qa_studies: synthetic correctness bench."""
    return _finite_blob(hwanung2_qa_studies.bench_hwanung2_qa_studies(seed))


def bench_jacheongbi2_qa_studies_family(seed: int = _SEED + 5):
    """jacheongbi2_qa_studies: synthetic correctness bench."""
    return _finite_blob(jacheongbi2_qa_studies.bench_jacheongbi2_qa_studies(seed))
