import pytest

from quant_fund.research import benches_w1935


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anchancho_qa_studies_family",
        "bench_jarjacha_qa_studies_family",
        "bench_kharisiri_qa_studies_family",
        "bench_muki_qa_studies_family",
        "bench_pishtaco_qa_studies_family",
        "bench_sirenito_qa_studies_family",
    ],
)
def test_benches_w1935(fam):
    out = getattr(benches_w1935, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
