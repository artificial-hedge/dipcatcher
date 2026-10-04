import pytest

from quant_fund.research import benches_w1462


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crab_qa_studies_family",
        "bench_jellyfish_qa_studies_family",
        "bench_octopus_qa_studies_family",
        "bench_seahorse_qa_studies_family",
        "bench_squid_qa_studies_family",
        "bench_stingray_qa_studies_family",
    ],
)
def test_benches_w1462(fam):
    out = getattr(benches_w1462, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
