import pytest

from quant_fund.research import benches_w1886


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bahamut_qa_studies_family",
        "bench_falak_qa_studies_family",
        "bench_ghoula_qa_studies_family",
        "bench_karkadann_qa_studies_family",
        "bench_nasnas_qa_studies_family",
        "bench_shahmaran_qa_studies_family",
    ],
)
def test_benches_w1886(fam):
    out = getattr(benches_w1886, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
