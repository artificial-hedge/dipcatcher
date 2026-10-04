import pytest

from quant_fund.research import benches_w1376


@pytest.mark.parametrize(
    "fam",
    [
        "bench_blender_bot_studies_family",
        "bench_conv_ai2_studies_family",
        "bench_daily_dialog_studies_family",
        "bench_dstc_lite_studies_family",
        "bench_empathy_dialog_studies_family",
        "bench_persona_chat_studies_family",
    ],
)
def test_benches_w1376(fam):
    out = getattr(benches_w1376, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
