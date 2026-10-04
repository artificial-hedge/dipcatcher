import pytest

from quant_fund.research import benches_w1440


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bamboo_qa_studies_family",
        "bench_cactus_qa_studies_family",
        "bench_fern_qa_studies_family",
        "bench_moss_qa_studies_family",
        "bench_pine_qa_studies_family",
        "bench_vine_qa_studies_family",
    ],
)
def test_benches_w1440(fam):
    out = getattr(benches_w1440, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
