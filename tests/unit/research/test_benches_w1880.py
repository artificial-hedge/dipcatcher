import pytest

from quant_fund.research import benches_w1880


@pytest.mark.parametrize(
    "fam",
    [
        "bench_achimi_qa_studies_family",
        "bench_iyezid_qa_studies_family",
        "bench_mazer_qa_studies_family",
        "bench_milkart_qa_studies_family",
        "bench_tamgak_qa_studies_family",
        "bench_tesfit_qa_studies_family",
    ],
)
def test_benches_w1880(fam):
    out = getattr(benches_w1880, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
