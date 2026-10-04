import pytest

from quant_fund.research import benches_w1624


@pytest.mark.parametrize(
    "fam",
    [
        "bench_arctic_hare_qa_studies_family",
        "bench_gyrfalcon_qa_studies_family",
        "bench_pallas_manul_qa_studies_family",
        "bench_ptarmigan_qa_studies_family",
        "bench_snowshoe_qa_studies_family",
        "bench_tundra_swan_qa_studies_family",
    ],
)
def test_benches_w1624(fam):
    out = getattr(benches_w1624, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
