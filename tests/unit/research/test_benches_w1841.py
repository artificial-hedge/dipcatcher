import pytest

from quant_fund.research import benches_w1841


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ashtar2_qa_studies_family",
        "bench_baalpeor_qa_studies_family",
        "bench_chemosh3_qa_studies_family",
        "bench_dibon2_qa_studies_family",
        "bench_kiriath2_qa_studies_family",
        "bench_nebo2_qa_studies_family",
    ],
)
def test_benches_w1841(fam):
    out = getattr(benches_w1841, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
