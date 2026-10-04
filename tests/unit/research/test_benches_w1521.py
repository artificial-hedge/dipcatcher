import pytest

from quant_fund.research import benches_w1521


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cattleya_qa_studies_family",
        "bench_cymbidium_qa_studies_family",
        "bench_dendrobium_qa_studies_family",
        "bench_oncidium_qa_studies_family",
        "bench_paphiopedilum_qa_studies_family",
        "bench_phalaenopsis_qa_studies_family",
    ],
)
def test_benches_w1521(fam):
    out = getattr(benches_w1521, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
