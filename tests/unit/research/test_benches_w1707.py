import pytest

from quant_fund.research import benches_w1707


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ahti_qa_studies_family",
        "bench_ilmatar_qa_studies_family",
        "bench_kiputytto_qa_studies_family",
        "bench_louhi_qa_studies_family",
        "bench_tapio_qa_studies_family",
        "bench_ukko_qa_studies_family",
    ],
)
def test_benches_w1707(fam):
    out = getattr(benches_w1707, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
