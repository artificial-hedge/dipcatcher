import pytest

from quant_fund.research import benches_w1516


@pytest.mark.parametrize(
    "fam",
    [
        "bench_atlas_moth_qa_studies_family",
        "bench_gypsy_moth_qa_studies_family",
        "bench_hawk_moth_qa_studies_family",
        "bench_luna_moth_qa_studies_family",
        "bench_tussock_moth_qa_studies_family",
        "bench_underwing_qa_studies_family",
    ],
)
def test_benches_w1516(fam):
    out = getattr(benches_w1516, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
