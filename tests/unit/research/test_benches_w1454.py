import pytest

from quant_fund.research import benches_w1454


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aphid_qa_studies_family",
        "bench_hornet_qa_studies_family",
        "bench_locust_qa_studies_family",
        "bench_mosquito_qa_studies_family",
        "bench_scarab_qa_studies_family",
        "bench_termite_qa_studies_family",
    ],
)
def test_benches_w1454(fam):
    out = getattr(benches_w1454, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
