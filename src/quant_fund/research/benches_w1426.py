"""Wave-1426 bench adapters: leisure canon (SYNTHETIC only)."""

from quant_fund.models import (
    challenge_qa_studies,
    contest_qa_studies,
    game_qa_studies,
    hobby_qa_studies,
    leisure_qa_studies,
    match_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14260


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_challenge_qa_studies_family(seed: int = _SEED + 0):
    """challenge_qa_studies: synthetic correctness bench."""
    return _finite_blob(challenge_qa_studies.bench_challenge_qa_studies(seed))


def bench_contest_qa_studies_family(seed: int = _SEED + 1):
    """contest_qa_studies: synthetic correctness bench."""
    return _finite_blob(contest_qa_studies.bench_contest_qa_studies(seed))


def bench_game_qa_studies_family(seed: int = _SEED + 2):
    """game_qa_studies: synthetic correctness bench."""
    return _finite_blob(game_qa_studies.bench_game_qa_studies(seed))


def bench_hobby_qa_studies_family(seed: int = _SEED + 3):
    """hobby_qa_studies: synthetic correctness bench."""
    return _finite_blob(hobby_qa_studies.bench_hobby_qa_studies(seed))


def bench_leisure_qa_studies_family(seed: int = _SEED + 4):
    """leisure_qa_studies: synthetic correctness bench."""
    return _finite_blob(leisure_qa_studies.bench_leisure_qa_studies(seed))


def bench_match_qa_studies_family(seed: int = _SEED + 5):
    """match_qa_studies: synthetic correctness bench."""
    return _finite_blob(match_qa_studies.bench_match_qa_studies(seed))
