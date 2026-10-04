import pytest

from quant_fund.research import benches_w1605


@pytest.mark.parametrize(
    "fam",
    [
        "bench_decorator_qa_studies_family",
        "bench_fiddler_qa_studies_family",
        "bench_rock_crab_qa_studies_family",
        "bench_sea_snake_qa_studies_family",
        "bench_skate_qa_studies_family",
        "bench_wobbegong_qa_studies_family",
    ],
)
def test_benches_w1605(fam):
    out = getattr(benches_w1605, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
