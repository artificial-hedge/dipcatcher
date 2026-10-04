import pytest

from quant_fund.research import benches_w1695


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dijiang_qa_studies_family",
        "bench_huli_qa_studies_family",
        "bench_jiangshi_qa_studies_family",
        "bench_mogwai_qa_studies_family",
        "bench_yaoguai_qa_studies_family",
        "bench_zhuyin_qa_studies_family",
    ],
)
def test_benches_w1695(fam):
    out = getattr(benches_w1695, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
