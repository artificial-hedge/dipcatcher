import pytest

from quant_fund.research import benches_w1879


@pytest.mark.parametrize(
    "fam",
    [
        "bench_afriye_qa_studies_family",
        "bench_almajira_qa_studies_family",
        "bench_djinnet_qa_studies_family",
        "bench_kel_essuf_qa_studies_family",
        "bench_tanit_lok_qa_studies_family",
        "bench_tin_hinan_qa_studies_family",
    ],
)
def test_benches_w1879(fam):
    out = getattr(benches_w1879, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
