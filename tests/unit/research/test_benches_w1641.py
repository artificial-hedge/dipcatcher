import pytest

from quant_fund.research import benches_w1641


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bixie_qa_studies_family",
        "bench_fenghuang_qa_studies_family",
        "bench_hundun_qa_studies_family",
        "bench_qiongqi_qa_studies_family",
        "bench_taotie_qa_studies_family",
        "bench_taowu_qa_studies_family",
    ],
)
def test_benches_w1641(fam):
    out = getattr(benches_w1641, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
