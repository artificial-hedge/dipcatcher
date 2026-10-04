import pytest

from quant_fund.research import benches_w1894


@pytest.mark.parametrize(
    "fam",
    [
        "bench_asakku_qa_studies_family",
        "bench_ekimmu_qa_studies_family",
        "bench_etemmu_qa_studies_family",
        "bench_gidim_qa_studies_family",
        "bench_maskim_qa_studies_family",
        "bench_sebettu_qa_studies_family",
    ],
)
def test_benches_w1894(fam):
    out = getattr(benches_w1894, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
