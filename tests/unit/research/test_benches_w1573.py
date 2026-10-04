import pytest

from quant_fund.research import benches_w1573


@pytest.mark.parametrize(
    "fam",
    [
        "bench_addax_qa_studies_family",
        "bench_fennec_qa_studies_family",
        "bench_jerboa_qa_studies_family",
        "bench_meerkat_qa_studies_family",
        "bench_onager_qa_studies_family",
        "bench_pangolin_qa_studies_family",
    ],
)
def test_benches_w1573(fam):
    out = getattr(benches_w1573, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
