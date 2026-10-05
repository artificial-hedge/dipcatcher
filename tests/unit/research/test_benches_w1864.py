import pytest

from quant_fund.research import benches_w1864


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bors_qa_studies_family",
        "bench_culhwch_qa_studies_family",
        "bench_dinadan_qa_studies_family",
        "bench_palamedes_qa_studies_family",
        "bench_safir_qa_studies_family",
        "bench_segwarides_qa_studies_family",
    ],
)
def test_benches_w1864(fam):
    out = getattr(benches_w1864, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
