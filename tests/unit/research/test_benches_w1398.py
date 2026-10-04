import pytest

from quant_fund.research import benches_w1398


@pytest.mark.parametrize(
    "fam",
    [
        "bench_doc2dial_studies_family",
        "bench_finqa_lite_studies_family",
        "bench_hybridqa_lite_studies_family",
        "bench_infotabs_studies_family",
        "bench_ottqa_lite_studies_family",
        "bench_tab_cwq_studies_family",
    ],
)
def test_benches_w1398(fam):
    out = getattr(benches_w1398, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
