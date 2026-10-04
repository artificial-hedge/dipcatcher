import pytest

from quant_fund.research import benches_w1587


@pytest.mark.parametrize(
    "fam",
    [
        "bench_barasingha_qa_studies_family",
        "bench_brocket_qa_studies_family",
        "bench_huemul_qa_studies_family",
        "bench_mule_deer_qa_studies_family",
        "bench_sambar_qa_studies_family",
        "bench_taruca_qa_studies_family",
    ],
)
def test_benches_w1587(fam):
    out = getattr(benches_w1587, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
