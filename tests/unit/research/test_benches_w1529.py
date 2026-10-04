import pytest

from quant_fund.research import benches_w1529


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aquamarine_qa_studies_family",
        "bench_garnet_qa_studies_family",
        "bench_opal_qa_studies_family",
        "bench_ruby_qa_studies_family",
        "bench_tanzanite_qa_studies_family",
        "bench_tourmaline_qa_studies_family",
    ],
)
def test_benches_w1529(fam):
    out = getattr(benches_w1529, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
