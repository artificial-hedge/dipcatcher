import pytest

from quant_fund.research import benches_w1524


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bulrush_qa_studies_family",
        "bench_carex_qa_studies_family",
        "bench_cattail_qa_studies_family",
        "bench_cottongrass_qa_studies_family",
        "bench_reed_qa_studies_family",
        "bench_rush_qa_studies_family",
    ],
)
def test_benches_w1524(fam):
    out = getattr(benches_w1524, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
