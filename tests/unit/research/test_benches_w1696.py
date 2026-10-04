import pytest

from quant_fund.research import benches_w1696


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dongwanggong_qa_studies_family",
        "bench_fuxi_qa_studies_family",
        "bench_kuafu_qa_studies_family",
        "bench_nuwa_qa_studies_family",
        "bench_shennong_qa_studies_family",
        "bench_xiwangmu_qa_studies_family",
    ],
)
def test_benches_w1696(fam):
    out = getattr(benches_w1696, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
