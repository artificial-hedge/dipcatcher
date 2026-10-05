import pytest

from quant_fund.research import benches_w1826


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alkarisi2_qa_studies_family",
        "bench_baiyz2_qa_studies_family",
        "bench_erlik2_qa_studies_family",
        "bench_kydyr2_qa_studies_family",
        "bench_tenger2_qa_studies_family",
        "bench_ulgen2_qa_studies_family",
    ],
)
def test_benches_w1826(fam):
    out = getattr(benches_w1826, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
