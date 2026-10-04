import pytest

from quant_fund.research import benches_w1568


@pytest.mark.parametrize(
    "fam",
    [
        "bench_conger_qa_studies_family",
        "bench_garden_eel_qa_studies_family",
        "bench_hagfish_qa_studies_family",
        "bench_lamprey_qa_studies_family",
        "bench_moray_qa_studies_family",
        "bench_ribbon_eel_qa_studies_family",
    ],
)
def test_benches_w1568(fam):
    out = getattr(benches_w1568, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
