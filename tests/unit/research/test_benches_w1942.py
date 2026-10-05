import pytest

from quant_fund.research import benches_w1942


@pytest.mark.parametrize(
    "fam",
    [
        "bench_co_hon_qa_studies_family",
        "bench_hon_ma_qa_studies_family",
        "bench_ngu_tinh_qa_studies_family",
        "bench_quy_am_qa_studies_family",
        "bench_tinh_linh_qa_studies_family",
        "bench_yeu_quai_qa_studies_family",
    ],
)
def test_benches_w1942(fam):
    out = getattr(benches_w1942, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
