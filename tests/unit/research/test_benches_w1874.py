import pytest

from quant_fund.research import benches_w1874


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arkan_sonney_qa_studies_family",
        "bench_dozmary_qa_studies_family",
        "bench_loaghtan_qa_studies_family",
        "bench_shooil_ghoul_qa_studies_family",
        "bench_sleih_beggey_qa_studies_family",
        "bench_ushtey_qa_studies_family",
    ],
)
def test_benches_w1874(fam):
    out = getattr(benches_w1874, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
