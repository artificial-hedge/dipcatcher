import pytest

from quant_fund.research import benches_w1923


@pytest.mark.parametrize(
    "fam",
    [
        "bench_baigujing_qa_studies_family",
        "bench_hanba_qa_studies_family",
        "bench_jiuying_qa_studies_family",
        "bench_nian_qa_studies_family",
        "bench_wuzhiqi_qa_studies_family",
        "bench_xiangliu_qa_studies_family",
    ],
)
def test_benches_w1923(fam):
    out = getattr(benches_w1923, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
