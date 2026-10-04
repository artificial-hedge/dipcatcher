"""Wave-1731 bench adapters: taino-myth canon (SYNTHETIC only)."""

from quant_fund.models import (
    atabei_qa_studies,
    boinayel_qa_studies,
    deminan_qa_studies,
    juracan_qa_studies,
    karacarol_qa_studies,
    yucahu_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 17310


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_atabei_qa_studies_family(seed: int = _SEED + 0):
    """atabei_qa_studies: synthetic correctness bench."""
    return _finite_blob(atabei_qa_studies.bench_atabei_qa_studies(seed))


def bench_boinayel_qa_studies_family(seed: int = _SEED + 1):
    """boinayel_qa_studies: synthetic correctness bench."""
    return _finite_blob(boinayel_qa_studies.bench_boinayel_qa_studies(seed))


def bench_deminan_qa_studies_family(seed: int = _SEED + 2):
    """deminan_qa_studies: synthetic correctness bench."""
    return _finite_blob(deminan_qa_studies.bench_deminan_qa_studies(seed))


def bench_juracan_qa_studies_family(seed: int = _SEED + 3):
    """juracan_qa_studies: synthetic correctness bench."""
    return _finite_blob(juracan_qa_studies.bench_juracan_qa_studies(seed))


def bench_karacarol_qa_studies_family(seed: int = _SEED + 4):
    """karacarol_qa_studies: synthetic correctness bench."""
    return _finite_blob(karacarol_qa_studies.bench_karacarol_qa_studies(seed))


def bench_yucahu_qa_studies_family(seed: int = _SEED + 5):
    """yucahu_qa_studies: synthetic correctness bench."""
    return _finite_blob(yucahu_qa_studies.bench_yucahu_qa_studies(seed))
