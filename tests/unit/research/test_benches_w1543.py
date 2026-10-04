import pytest

from quant_fund.research import benches_w1543


@pytest.mark.parametrize(
    "fam",
    [
        "bench_oilbird_qa_studies_family",
        "bench_owlet_nightjar_qa_studies_family",
        "bench_pauraque_qa_studies_family",
        "bench_poorwill_qa_studies_family",
        "bench_potoo_qa_studies_family",
        "bench_whip_poor_will_qa_studies_family",
    ],
)
def test_benches_w1543(fam):
    out = getattr(benches_w1543, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
