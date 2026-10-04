import pytest

from quant_fund.research import benches_w1464


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abyss_qa_studies_family",
        "bench_beacon_qa_studies_family",
        "bench_blizzard_qa_studies_family",
        "bench_monolith_qa_studies_family",
        "bench_spire_qa_studies_family",
        "bench_tempest_qa_studies_family",
    ],
)
def test_benches_w1464(fam):
    out = getattr(benches_w1464, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
