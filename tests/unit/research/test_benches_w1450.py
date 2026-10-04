import pytest

from quant_fund.research import benches_w1450


@pytest.mark.parametrize(
    "fam",
    [
        "bench_barn_qa_studies_family",
        "bench_cow_qa_studies_family",
        "bench_goat_qa_studies_family",
        "bench_horse_qa_studies_family",
        "bench_pig_qa_studies_family",
        "bench_sheep_qa_studies_family",
    ],
)
def test_benches_w1450(fam):
    out = getattr(benches_w1450, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
