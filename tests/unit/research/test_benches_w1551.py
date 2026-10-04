import pytest

from quant_fund.research import benches_w1551


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bushmaster_qa_studies_family",
        "bench_copperhead_qa_studies_family",
        "bench_coral_snake_qa_studies_family",
        "bench_cottonmouth_qa_studies_family",
        "bench_fer_de_lance_qa_studies_family",
        "bench_rattlesnake_qa_studies_family",
    ],
)
def test_benches_w1551(fam):
    out = getattr(benches_w1551, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
