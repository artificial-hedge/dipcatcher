import pytest

from quant_fund.research import benches_w1926


@pytest.mark.parametrize(
    "fam",
    [
        "bench_albasty_qa_studies_family",
        "bench_albi_qa_studies_family",
        "bench_chinka_qa_studies_family",
        "bench_furts_qa_studies_family",
        "bench_gorgogosh_qa_studies_family",
        "bench_rukhi_qa_studies_family",
    ],
)
def test_benches_w1926(fam):
    out = getattr(benches_w1926, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
