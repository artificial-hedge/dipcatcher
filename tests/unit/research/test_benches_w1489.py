import pytest

from quant_fund.research import benches_w1489


@pytest.mark.parametrize(
    "fam",
    [
        "bench_clover_qa_studies_family",
        "bench_heather_qa_studies_family",
        "bench_lavender_qa_studies_family",
        "bench_lilac_qa_studies_family",
        "bench_marigold_qa_studies_family",
        "bench_primrose_qa_studies_family",
    ],
)
def test_benches_w1489(fam):
    out = getattr(benches_w1489, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
