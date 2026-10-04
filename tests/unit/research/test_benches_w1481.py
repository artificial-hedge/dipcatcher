import pytest

from quant_fund.research import benches_w1481


@pytest.mark.parametrize(
    "fam",
    [
        "bench_earwig_qa_studies_family",
        "bench_katydid_qa_studies_family",
        "bench_mayfly_qa_studies_family",
        "bench_stonefly_qa_studies_family",
        "bench_wasp_qa_studies_family",
        "bench_weevil_qa_studies_family",
    ],
)
def test_benches_w1481(fam):
    out = getattr(benches_w1481, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
