import pytest

from quant_fund.research import benches_w1613


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aurochs_qa_studies_family",
        "bench_banteng_qa_studies_family",
        "bench_gaur_qa_studies_family",
        "bench_saola_qa_studies_family",
        "bench_tamaraw_qa_studies_family",
        "bench_yak_qa_studies_family",
    ],
)
def test_benches_w1613(fam):
    out = getattr(benches_w1613, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
