import pytest

from quant_fund.research import benches_w1437


@pytest.mark.parametrize(
    "fam",
    [
        "bench_deity_qa_studies_family",
        "bench_dragon_qa_studies_family",
        "bench_hero_qa_studies_family",
        "bench_olympus_qa_studies_family",
        "bench_phoenix_qa_studies_family",
        "bench_titan_qa_studies_family",
    ],
)
def test_benches_w1437(fam):
    out = getattr(benches_w1437, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
