import pytest

from quant_fund.research import benches_w1779


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fuxi_qa_studies_family",
        "bench_huangdi_qa_studies_family",
        "bench_nuwa_qa_studies_family",
        "bench_shennong_qa_studies_family",
        "bench_xihe_qa_studies_family",
        "bench_yandi_qa_studies_family",
    ],
)
def test_benches_w1779(fam):
    out = getattr(benches_w1779, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
