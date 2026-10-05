import pytest

from quant_fund.research import benches_w1877


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amayya_qa_studies_family",
        "bench_atete_qa_studies_family",
        "bench_guzil_qa_studies_family",
        "bench_igal_qa_studies_family",
        "bench_tiniri_qa_studies_family",
        "bench_warpon_qa_studies_family",
    ],
)
def test_benches_w1877(fam):
    out = getattr(benches_w1877, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
