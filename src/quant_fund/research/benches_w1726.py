"""Wave-1726 bench adapters: korean-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    dalnim_qa_studies,
    dangun_qa_studies,
    haenim_qa_studies,
    hwanin_qa_studies,
    hwanung_qa_studies,
    samshin_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17260


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_dalnim_qa_studies_family(seed: int = _SEED + 0):
    """dalnim_qa_studies: synthetic correctness bench."""
    return _finite_blob(dalnim_qa_studies.bench_dalnim_qa_studies(seed))


def bench_dangun_qa_studies_family(seed: int = _SEED + 1):
    """dangun_qa_studies: synthetic correctness bench."""
    return _finite_blob(dangun_qa_studies.bench_dangun_qa_studies(seed))


def bench_haenim_qa_studies_family(seed: int = _SEED + 2):
    """haenim_qa_studies: synthetic correctness bench."""
    return _finite_blob(haenim_qa_studies.bench_haenim_qa_studies(seed))


def bench_hwanin_qa_studies_family(seed: int = _SEED + 3):
    """hwanin_qa_studies: synthetic correctness bench."""
    return _finite_blob(hwanin_qa_studies.bench_hwanin_qa_studies(seed))


def bench_hwanung_qa_studies_family(seed: int = _SEED + 4):
    """hwanung_qa_studies: synthetic correctness bench."""
    return _finite_blob(hwanung_qa_studies.bench_hwanung_qa_studies(seed))


def bench_samshin_qa_studies_family(seed: int = _SEED + 5):
    """samshin_qa_studies: synthetic correctness bench."""
    return _finite_blob(samshin_qa_studies.bench_samshin_qa_studies(seed))
