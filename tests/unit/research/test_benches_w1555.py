import pytest

from quant_fund.research import benches_w1555


@pytest.mark.parametrize(
    "fam",
    [
        "bench_harvestman_qa_studies_family",
        "bench_pseudoscorpion_qa_studies_family",
        "bench_solifuge_qa_studies_family",
        "bench_tick_qa_studies_family",
        "bench_vinegaroon_qa_studies_family",
        "bench_whip_scorpion_qa_studies_family",
    ],
)
def test_benches_w1555(fam):
    out = getattr(benches_w1555, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
