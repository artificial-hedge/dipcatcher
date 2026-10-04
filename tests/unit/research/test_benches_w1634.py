import pytest

from quant_fund.research import benches_w1634


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gnome_2_qa_studies_family",
        "bench_ifrit_qa_studies_family",
        "bench_marid_qa_studies_family",
        "bench_salamander_2_qa_studies_family",
        "bench_sylph_2_qa_studies_family",
        "bench_undine_qa_studies_family",
    ],
)
def test_benches_w1634(fam):
    out = getattr(benches_w1634, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
