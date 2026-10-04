import pytest

from quant_fund.research import benches_w1777


@pytest.mark.parametrize(
    "fam",
    [
        "bench_barong_qa_studies_family",
        "bench_garuda_qa_studies_family",
        "bench_nyai_qa_studies_family",
        "bench_raksasa_qa_studies_family",
        "bench_rangda_qa_studies_family",
        "bench_semar_qa_studies_family",
    ],
)
def test_benches_w1777(fam):
    out = getattr(benches_w1777, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
