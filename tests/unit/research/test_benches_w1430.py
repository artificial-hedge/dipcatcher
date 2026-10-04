import pytest

from quant_fund.research import benches_w1430


@pytest.mark.parametrize(
    "fam",
    [
        "bench_animal_qa_studies_family",
        "bench_bird_qa_studies_family",
        "bench_ecosystem_qa_studies_family",
        "bench_fish_qa_studies_family",
        "bench_habitat_qa_studies_family",
        "bench_insect_qa_studies_family",
    ],
)
def test_benches_w1430(fam):
    out = getattr(benches_w1430, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
