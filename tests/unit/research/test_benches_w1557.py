import pytest

from quant_fund.research import benches_w1557


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arapaima_qa_studies_family",
        "bench_electric_eel_qa_studies_family",
        "bench_knifefish_qa_studies_family",
        "bench_oscar_qa_studies_family",
        "bench_pacu_qa_studies_family",
        "bench_tetra_qa_studies_family",
    ],
)
def test_benches_w1557(fam):
    out = getattr(benches_w1557, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
