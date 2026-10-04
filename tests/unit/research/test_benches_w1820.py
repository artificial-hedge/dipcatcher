import pytest

from quant_fund.research import benches_w1820


@pytest.mark.parametrize(
    "fam",
    [
        "bench_changyi2_qa_studies_family",
        "bench_gun2_qa_studies_family",
        "bench_shennong2_qa_studies_family",
        "bench_xihe2_qa_studies_family",
        "bench_yandi2_qa_studies_family",
        "bench_yaoji2_qa_studies_family",
    ],
)
def test_benches_w1820(fam):
    out = getattr(benches_w1820, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
