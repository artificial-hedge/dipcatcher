import pytest

from quant_fund.research import benches_w1931


@pytest.mark.parametrize(
    "fam",
    [
        "bench_deer_woman_qa_studies_family",
        "bench_kachina_qa_studies_family",
        "bench_manitou_qa_studies_family",
        "bench_mishipeshu_qa_studies_family",
        "bench_naagloshii_qa_studies_family",
        "bench_pukwudgie_qa_studies_family",
    ],
)
def test_benches_w1931(fam):
    out = getattr(benches_w1931, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
