import pytest

from quant_fund.research import benches_w1354


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arc_easy2_studies_family",
        "bench_boolq_lite_studies_family",
        "bench_cosmos_qa_studies_family",
        "bench_race_lite_studies_family",
        "bench_sciq_lite_studies_family",
        "bench_social_qa_studies_family",
    ],
)
def test_benches_w1354(fam):
    out = getattr(benches_w1354, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
