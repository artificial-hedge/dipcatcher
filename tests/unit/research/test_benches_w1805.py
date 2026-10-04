import pytest

from quant_fund.research import benches_w1805


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bereginia2_qa_studies_family",
        "bench_bogdan2_qa_studies_family",
        "bench_kupalo2_qa_studies_family",
        "bench_radegast2_qa_studies_family",
        "bench_rod2_qa_studies_family",
        "bench_ziva2_qa_studies_family",
    ],
)
def test_benches_w1805(fam):
    out = getattr(benches_w1805, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
