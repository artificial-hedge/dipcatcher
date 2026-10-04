import pytest

from quant_fund.research import benches_w1444


@pytest.mark.parametrize(
    "fam",
    [
        "bench_brook_qa_studies_family",
        "bench_creek_qa_studies_family",
        "bench_delta_qa_studies_family",
        "bench_estuary_qa_studies_family",
        "bench_marsh_qa_studies_family",
        "bench_pond_qa_studies_family",
    ],
)
def test_benches_w1444(fam):
    out = getattr(benches_w1444, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
