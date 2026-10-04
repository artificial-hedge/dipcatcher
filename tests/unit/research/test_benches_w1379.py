import pytest

from quant_fund.research import benches_w1379


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bary_score_studies_family",
        "bench_cider_lite_studies_family",
        "bench_gleu_lite_studies_family",
        "bench_kl_div_eval_studies_family",
        "bench_rouge_we_studies_family",
        "bench_wmt_metric_studies_family",
    ],
)
def test_benches_w1379(fam):
    out = getattr(benches_w1379, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
