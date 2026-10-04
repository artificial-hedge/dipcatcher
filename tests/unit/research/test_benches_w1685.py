import pytest

from quant_fund.research import benches_w1685


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ettin_qa_studies_family",
        "bench_fylgja_qa_studies_family",
        "bench_landvaettir_qa_studies_family",
        "bench_nokken_qa_studies_family",
        "bench_seidr_qa_studies_family",
        "bench_vette_qa_studies_family",
    ],
)
def test_benches_w1685(fam):
    out = getattr(benches_w1685, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
