import pytest

from quant_fund.research import benches_w1871


@pytest.mark.parametrize(
    "fam",
    [
        "bench_buggane_qa_studies_family",
        "bench_fenodyree_qa_studies_family",
        "bench_glashtyn_qa_studies_family",
        "bench_moddey_dhoo_qa_studies_family",
        "bench_phynnodderee_qa_studies_family",
        "bench_tarroo_ushtey_qa_studies_family",
    ],
)
def test_benches_w1871(fam):
    out = getattr(benches_w1871, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
