import pytest

from quant_fund.research import benches_w1618


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anglerfish_qa_studies_family",
        "bench_bristlemouth_qa_studies_family",
        "bench_grenadier_qa_studies_family",
        "bench_hatchetfish_qa_studies_family",
        "bench_lanternfish_qa_studies_family",
        "bench_viperfish_qa_studies_family",
    ],
)
def test_benches_w1618(fam):
    out = getattr(benches_w1618, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
