import pytest

from quant_fund.research import benches_w1360


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bioasq_lite_studies_family",
        "bench_cite_worth_studies_family",
        "bench_climate_fever_studies_family",
        "bench_fever_lite_studies_family",
        "bench_touch_e_studies_family",
        "bench_verdict_qa_studies_family",
    ],
)
def test_benches_w1360(fam):
    out = getattr(benches_w1360, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
