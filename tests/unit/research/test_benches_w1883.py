import pytest

from quant_fund.research import benches_w1883


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aewan_qa_studies_family",
        "bench_banguilet_qa_studies_family",
        "bench_hemmi_qa_studies_family",
        "bench_maziun_qa_studies_family",
        "bench_tissardal_qa_studies_family",
        "bench_zilalsen_qa_studies_family",
    ],
)
def test_benches_w1883(fam):
    out = getattr(benches_w1883, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
