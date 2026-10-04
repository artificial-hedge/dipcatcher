import pytest

from quant_fund.research import benches_w1586


@pytest.mark.parametrize(
    "fam",
    [
        "bench_beira_qa_studies_family",
        "bench_gemsbok_qa_studies_family",
        "bench_madoqua_qa_studies_family",
        "bench_oribi_qa_studies_family",
        "bench_reedbuck_qa_studies_family",
        "bench_tsessebe_qa_studies_family",
    ],
)
def test_benches_w1586(fam):
    out = getattr(benches_w1586, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
