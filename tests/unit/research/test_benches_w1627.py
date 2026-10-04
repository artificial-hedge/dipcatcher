import pytest

from quant_fund.research import benches_w1627


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cave_crayfish_qa_studies_family",
        "bench_cave_scorpion_qa_studies_family",
        "bench_cave_springtail_qa_studies_family",
        "bench_cave_worm_qa_studies_family",
        "bench_stygobite_qa_studies_family",
        "bench_troglofish_qa_studies_family",
    ],
)
def test_benches_w1627(fam):
    out = getattr(benches_w1627, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
