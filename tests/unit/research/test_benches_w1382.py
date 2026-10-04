import pytest

from quant_fund.research import benches_w1382


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fact_score_studies_family",
        "bench_gpt_score_studies_family",
        "bench_helm_lite_studies_family",
        "bench_lmsys_eval_studies_family",
        "bench_nugget_eval_studies_family",
        "bench_vicuna_bench_studies_family",
    ],
)
def test_benches_w1382(fam):
    out = getattr(benches_w1382, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
