import pytest

from quant_fund.research import benches_w1558


@pytest.mark.parametrize(
    "fam",
    [
        "bench_char_qa_studies_family",
        "bench_dolly_varden_qa_studies_family",
        "bench_grayling_qa_studies_family",
        "bench_sockeye_qa_studies_family",
        "bench_steelhead_qa_studies_family",
        "bench_whitefish_qa_studies_family",
    ],
)
def test_benches_w1558(fam):
    out = getattr(benches_w1558, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
