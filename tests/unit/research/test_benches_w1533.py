import pytest

from quant_fund.research import benches_w1533


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aracari_qa_studies_family",
        "bench_barbet_qa_studies_family",
        "bench_honeyguide_qa_studies_family",
        "bench_hornbill_qa_studies_family",
        "bench_quetzal_qa_studies_family",
        "bench_trogon_qa_studies_family",
    ],
)
def test_benches_w1533(fam):
    out = getattr(benches_w1533, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
