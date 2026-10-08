"""Wave-1376 bench adapters: dialogue-system canon (SYNTHETIC only)."""

from quant_fund.models import (
    blender_bot_studies,
    conv_ai2_studies,
    daily_dialog_studies,
    dstc_lite_studies,
    empathy_dialog_studies,
    persona_chat_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 13760


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_blender_bot_studies_family(seed: int = _SEED + 0):
    """blender_bot_studies: synthetic correctness bench."""
    return _finite_blob(blender_bot_studies.bench_blender_bot_studies(seed))


def bench_conv_ai2_studies_family(seed: int = _SEED + 1):
    """conv_ai2_studies: synthetic correctness bench."""
    return _finite_blob(conv_ai2_studies.bench_conv_ai2_studies(seed))


def bench_daily_dialog_studies_family(seed: int = _SEED + 2):
    """daily_dialog_studies: synthetic correctness bench."""
    return _finite_blob(daily_dialog_studies.bench_daily_dialog_studies(seed))


def bench_dstc_lite_studies_family(seed: int = _SEED + 3):
    """dstc_lite_studies: synthetic correctness bench."""
    return _finite_blob(dstc_lite_studies.bench_dstc_lite_studies(seed))


def bench_empathy_dialog_studies_family(seed: int = _SEED + 4):
    """empathy_dialog_studies: synthetic correctness bench."""
    return _finite_blob(empathy_dialog_studies.bench_empathy_dialog_studies(seed))


def bench_persona_chat_studies_family(seed: int = _SEED + 5):
    """persona_chat_studies: synthetic correctness bench."""
    return _finite_blob(persona_chat_studies.bench_persona_chat_studies(seed))
