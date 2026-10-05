import pytest

from quant_fund.research import benches_w1891


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bergelmir_qa_studies_family",
        "bench_jormunrek_qa_studies_family",
        "bench_khan_tengri_qa_studies_family",
        "bench_peri_qa_studies_family",
        "bench_umm_sibyan_qa_studies_family",
        "bench_ymir_hrimthurs_qa_studies_family",
    ],
)
def test_benches_w1891(fam):
    out = getattr(benches_w1891, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
