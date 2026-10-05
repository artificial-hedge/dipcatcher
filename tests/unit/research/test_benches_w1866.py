import pytest

from quant_fund.research import benches_w1866


@pytest.mark.parametrize(
    "fam",
    [
        "bench_avilion_qa_studies_family",
        "bench_balan_qa_studies_family",
        "bench_ector_qa_studies_family",
        "bench_hector_cameliard_qa_studies_family",
        "bench_seneschal_qa_studies_family",
        "bench_ynis_qa_studies_family",
    ],
)
def test_benches_w1866(fam):
    out = getattr(benches_w1866, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
