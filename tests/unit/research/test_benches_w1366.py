import pytest

from quant_fund.research import benches_w1366


@pytest.mark.parametrize(
    "fam",
    [
        "bench_align_score_studies_family",
        "bench_dice_eval_studies_family",
        "bench_factcc_lite_studies_family",
        "bench_faith_eval_studies_family",
        "bench_quest_eval_studies_family",
        "bench_summa_eval_studies_family",
    ],
)
def test_benches_w1366(fam):
    out = getattr(benches_w1366, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
