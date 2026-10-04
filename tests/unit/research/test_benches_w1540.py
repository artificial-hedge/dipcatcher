import pytest

from quant_fund.research import benches_w1540


@pytest.mark.parametrize(
    "fam",
    [
        "bench_cattle_egret_qa_studies_family",
        "bench_glossy_ibis_qa_studies_family",
        "bench_great_egret_qa_studies_family",
        "bench_sacred_ibis_qa_studies_family",
        "bench_snowy_egret_qa_studies_family",
        "bench_squacco_qa_studies_family",
    ],
)
def test_benches_w1540(fam):
    out = getattr(benches_w1540, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
