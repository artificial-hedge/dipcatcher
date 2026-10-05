import pytest

from quant_fund.research import benches_w1798


@pytest.mark.parametrize(
    "fam",
    [
        "bench_changxi2_qa_studies_family",
        "bench_fuxi2_qa_studies_family",
        "bench_gonggong2_qa_studies_family",
        "bench_nuwa2_qa_studies_family",
        "bench_shennong2_qa_studies_family",
        "bench_zhurong2_qa_studies_family",
    ],
)
def test_benches_w1798(fam):
    out = getattr(benches_w1798, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
