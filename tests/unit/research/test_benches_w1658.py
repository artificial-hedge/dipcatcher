import pytest

from quant_fund.research import benches_w1658


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ahuizotl_qa_studies_family",
        "bench_alicanto_qa_studies_family",
        "bench_cadejo_qa_studies_family",
        "bench_cipactli_qa_studies_family",
        "bench_jinn_qa_studies_family",
        "bench_quetzalcoat_qa_studies_family",
    ],
)
def test_benches_w1658(fam):
    out = getattr(benches_w1658, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
