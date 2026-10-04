import pytest

from quant_fund.research import benches_w1536


@pytest.mark.parametrize(
    "fam",
    [
        "bench_collared_dove_qa_studies_family",
        "bench_dove_qa_studies_family",
        "bench_mourning_dove_qa_studies_family",
        "bench_pigeon_qa_studies_family",
        "bench_turtle_dove_qa_studies_family",
        "bench_woodpigeon_qa_studies_family",
    ],
)
def test_benches_w1536(fam):
    out = getattr(benches_w1536, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
