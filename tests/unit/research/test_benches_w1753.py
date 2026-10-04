import pytest

from quant_fund.research import benches_w1753


@pytest.mark.parametrize(
    "fam",
    [
        "bench_changxi_qa_studies_family",
        "bench_chiyou_qa_studies_family",
        "bench_gonggong_qa_studies_family",
        "bench_xihe_qa_studies_family",
        "bench_yinglong_qa_studies_family",
        "bench_zhurong_qa_studies_family",
    ],
)
def test_benches_w1753(fam):
    out = getattr(benches_w1753, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
