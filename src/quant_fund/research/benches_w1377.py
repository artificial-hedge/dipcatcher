"""Wave-1377 bench adapters: NLU-exotics canon (SYNTHETIC only)."""

from quant_fund.models import (
    abduction_lite_studies,
    board_game_qa_studies,
    conv_finqa_studies,
    dream_lite_studies,
    equiv_lite_studies,
    wsc_lite_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13770


def _finite_blob(blob):
    assert isinstance(blob, dict) and blob
    for k, v in blob.items():
        assert k.startswith("synthetic_"), k
        assert k not in _FORBIDDEN, k
        assert isinstance(v, float) and 0.0 <= v <= 1.0, (k, v)
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_abduction_lite_studies_family(seed: int = _SEED + 0):
    """abduction_lite_studies: synthetic correctness bench."""
    return _finite_blob(abduction_lite_studies.bench_abduction_lite_studies(seed))


def bench_board_game_qa_studies_family(seed: int = _SEED + 1):
    """board_game_qa_studies: synthetic correctness bench."""
    return _finite_blob(board_game_qa_studies.bench_board_game_qa_studies(seed))


def bench_conv_finqa_studies_family(seed: int = _SEED + 2):
    """conv_finqa_studies: synthetic correctness bench."""
    return _finite_blob(conv_finqa_studies.bench_conv_finqa_studies(seed))


def bench_dream_lite_studies_family(seed: int = _SEED + 3):
    """dream_lite_studies: synthetic correctness bench."""
    return _finite_blob(dream_lite_studies.bench_dream_lite_studies(seed))


def bench_equiv_lite_studies_family(seed: int = _SEED + 4):
    """equiv_lite_studies: synthetic correctness bench."""
    return _finite_blob(equiv_lite_studies.bench_equiv_lite_studies(seed))


def bench_wsc_lite_studies_family(seed: int = _SEED + 5):
    """wsc_lite_studies: synthetic correctness bench."""
    return _finite_blob(wsc_lite_studies.bench_wsc_lite_studies(seed))
