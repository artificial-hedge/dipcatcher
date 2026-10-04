import pytest

from quant_fund.research import benches_w1604


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dassie_qa_studies_family",
        "bench_gopher_qa_studies_family",
        "bench_mole_qa_studies_family",
        "bench_rabbit_qa_studies_family",
        "bench_shrew_qa_studies_family",
        "bench_springhare_qa_studies_family",
    ],
)
def test_benches_w1604(fam):
    out = getattr(benches_w1604, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
