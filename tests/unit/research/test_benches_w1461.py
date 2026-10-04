import pytest

from quant_fund.research import benches_w1461


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gorilla_qa_studies_family",
        "bench_jaguar_qa_studies_family",
        "bench_macaw_qa_studies_family",
        "bench_orangutan_qa_studies_family",
        "bench_sloth_qa_studies_family",
        "bench_toucan_qa_studies_family",
    ],
)
def test_benches_w1461(fam):
    out = getattr(benches_w1461, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
