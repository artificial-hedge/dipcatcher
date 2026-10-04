import pytest

from quant_fund.research import benches_w1377


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abduction_lite_studies_family",
        "bench_board_game_qa_studies_family",
        "bench_conv_finqa_studies_family",
        "bench_dream_lite_studies_family",
        "bench_equiv_lite_studies_family",
        "bench_wsc_lite_studies_family",
    ],
)
def test_benches_w1377(fam):
    out = getattr(benches_w1377, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
