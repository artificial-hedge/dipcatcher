import pytest

from quant_fund.research import benches_w1406


@pytest.mark.parametrize(
    "fam",
    [
        "bench_menat_qa_studies_family",
        "bench_syndq_lite_studies_family",
        "bench_teas_qa_studies_family",
        "bench_time_qa_studies_family",
        "bench_timedial_qa_studies_family",
        "bench_timetravel_lite_studies_family",
    ],
)
def test_benches_w1406(fam):
    out = getattr(benches_w1406, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
