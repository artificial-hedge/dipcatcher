import pytest

from quant_fund.research import benches_w1852


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aulisua_qa_studies_family",
        "bench_gurzil_qa_studies_family",
        "bench_iguc_qa_studies_family",
        "bench_lallus_qa_studies_family",
        "bench_macurgum_qa_studies_family",
        "bench_melyakina_qa_studies_family",
    ],
)
def test_benches_w1852(fam):
    out = getattr(benches_w1852, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
