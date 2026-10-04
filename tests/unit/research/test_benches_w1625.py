import pytest

from quant_fund.research import benches_w1625


@pytest.mark.parametrize(
    "fam",
    [
        "bench_blind_salamander_qa_studies_family",
        "bench_cave_shrimp_qa_studies_family",
        "bench_cave_spider_qa_studies_family",
        "bench_cave_swiftlet_qa_studies_family",
        "bench_grotto_salamander_qa_studies_family",
        "bench_proteus_qa_studies_family",
    ],
)
def test_benches_w1625(fam):
    out = getattr(benches_w1625, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
