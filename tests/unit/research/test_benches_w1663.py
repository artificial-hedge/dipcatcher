import pytest

from quant_fund.research import benches_w1663


@pytest.mark.parametrize(
    "fam",
    [
        "bench_barghest_qa_studies_family",
        "bench_black_dog_qa_studies_family",
        "bench_cat_sith_qa_studies_family",
        "bench_church_grim_qa_studies_family",
        "bench_cwn_annwn_qa_studies_family",
        "bench_grimalkin_qa_studies_family",
    ],
)
def test_benches_w1663(fam):
    out = getattr(benches_w1663, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
