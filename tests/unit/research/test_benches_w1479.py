import pytest

from quant_fund.research import benches_w1479


@pytest.mark.parametrize(
    "fam",
    [
        "bench_agouti_qa_studies_family",
        "bench_armadillo_qa_studies_family",
        "bench_capybara_qa_studies_family",
        "bench_coati_qa_studies_family",
        "bench_peccary_qa_studies_family",
        "bench_tapir_qa_studies_family",
    ],
)
def test_benches_w1479(fam):
    out = getattr(benches_w1479, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
