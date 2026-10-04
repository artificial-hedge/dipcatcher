import pytest

from quant_fund.research import benches_w1511


@pytest.mark.parametrize(
    "fam",
    [
        "bench_brilliant_qa_studies_family",
        "bench_hermit_qa_studies_family",
        "bench_hummingbird_qa_studies_family",
        "bench_sapphire_qa_studies_family",
        "bench_topaz_qa_studies_family",
        "bench_woodstar_qa_studies_family",
    ],
)
def test_benches_w1511(fam):
    out = getattr(benches_w1511, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
