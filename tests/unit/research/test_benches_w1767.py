import pytest

from quant_fund.research import benches_w1767


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coniraya_qa_studies_family",
        "bench_guanare_qa_studies_family",
        "bench_inti_qa_studies_family",
        "bench_pachacamac_qa_studies_family",
        "bench_supay_qa_studies_family",
        "bench_viracocha_qa_studies_family",
    ],
)
def test_benches_w1767(fam):
    out = getattr(benches_w1767, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
