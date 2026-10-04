import pytest

from quant_fund.research import benches_w1582


@pytest.mark.parametrize(
    "fam",
    [
        "bench_flying_fox_qa_studies_family",
        "bench_horseshoe_bat_qa_studies_family",
        "bench_leaf_nosed_qa_studies_family",
        "bench_noctule_qa_studies_family",
        "bench_pipistrelle_qa_studies_family",
        "bench_vampire_qa_studies_family",
    ],
)
def test_benches_w1582(fam):
    out = getattr(benches_w1582, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
