import pytest

from quant_fund.research import benches_w1653


@pytest.mark.parametrize(
    "fam",
    [
        "bench_alion_qa_studies_family",
        "bench_catoblepas_qa_studies_family",
        "bench_jasconius_qa_studies_family",
        "bench_pard_qa_studies_family",
        "bench_peluda_qa_studies_family",
        "bench_zaratan_qa_studies_family",
    ],
)
def test_benches_w1653(fam):
    out = getattr(benches_w1653, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
