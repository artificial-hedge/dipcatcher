import pytest

from quant_fund.research import benches_w1706


@pytest.mark.parametrize(
    "fam",
    [
        "bench_akana_qa_studies_family",
        "bench_erlik_qa_studies_family",
        "bench_kayra_qa_studies_family",
        "bench_perysh_qa_studies_family",
        "bench_tengri_qa_studies_family",
        "bench_ulgen_qa_studies_family",
    ],
)
def test_benches_w1706(fam):
    out = getattr(benches_w1706, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
