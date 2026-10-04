"""Wave-1362 bench adapters: fake-news canon (SYNTHETIC only)."""

from quant_fund.models import (
    check_that_studies,
    claim_buster_studies,
    emergent_lite_studies,
    fake_news_studies,
    snopes_lite_studies,
    stance_detect_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13620


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_check_that_studies_family(seed: int = _SEED + 0):
    """check_that_studies: synthetic correctness bench."""
    return _finite_blob(check_that_studies.bench_check_that_studies(seed))


def bench_claim_buster_studies_family(seed: int = _SEED + 1):
    """claim_buster_studies: synthetic correctness bench."""
    return _finite_blob(claim_buster_studies.bench_claim_buster_studies(seed))


def bench_emergent_lite_studies_family(seed: int = _SEED + 2):
    """emergent_lite_studies: synthetic correctness bench."""
    return _finite_blob(emergent_lite_studies.bench_emergent_lite_studies(seed))


def bench_fake_news_studies_family(seed: int = _SEED + 3):
    """fake_news_studies: synthetic correctness bench."""
    return _finite_blob(fake_news_studies.bench_fake_news_studies(seed))


def bench_snopes_lite_studies_family(seed: int = _SEED + 4):
    """snopes_lite_studies: synthetic correctness bench."""
    return _finite_blob(snopes_lite_studies.bench_snopes_lite_studies(seed))


def bench_stance_detect_studies_family(seed: int = _SEED + 5):
    """stance_detect_studies: synthetic correctness bench."""
    return _finite_blob(stance_detect_studies.bench_stance_detect_studies(seed))
