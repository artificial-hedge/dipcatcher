import pytest

from quant_fund.research import benches_w1332


@pytest.mark.parametrize(
    "fam",
    [
        "bench_decontaminate_studies_family",
        "bench_eval_bias_studies_family",
        "bench_fair_eval_studies_family",
        "bench_g_eval_studies_family",
        "bench_ngram_overlap_studies_family",
        "bench_pandalm_studies_family",
    ],
)
def test_benches_w1332(fam):
    out = getattr(benches_w1332, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
