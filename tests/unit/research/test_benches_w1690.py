import pytest

from quant_fund.research import benches_w1690


@pytest.mark.parametrize(
    "fam",
    [
        "bench_kladenets_qa_studies_family",
        "bench_kostroma_qa_studies_family",
        "bench_leshii_qa_studies_family",
        "bench_morozko_qa_studies_family",
        "bench_vedmak_qa_studies_family",
        "bench_yarilo_qa_studies_family",
    ],
)
def test_benches_w1690(fam):
    out = getattr(benches_w1690, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
