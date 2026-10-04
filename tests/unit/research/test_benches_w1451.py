import pytest

from quant_fund.research import benches_w1451


@pytest.mark.parametrize(
    "fam",
    [
        "bench_apple_qa_studies_family",
        "bench_cherry_qa_studies_family",
        "bench_grape_qa_studies_family",
        "bench_lemon_qa_studies_family",
        "bench_mango_qa_studies_family",
        "bench_peach_qa_studies_family",
    ],
)
def test_benches_w1451(fam):
    out = getattr(benches_w1451, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
