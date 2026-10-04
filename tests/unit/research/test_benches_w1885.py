import pytest

from quant_fund.research import benches_w1885


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amenokal_qa_studies_family",
        "bench_ammonion_qa_studies_family",
        "bench_atlas_deity_qa_studies_family",
        "bench_imajeghen_qa_studies_family",
        "bench_melqart_libya_qa_studies_family",
        "bench_tritogeneia_qa_studies_family",
    ],
)
def test_benches_w1885(fam):
    out = getattr(benches_w1885, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
