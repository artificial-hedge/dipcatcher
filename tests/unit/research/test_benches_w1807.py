import pytest

from quant_fund.research import benches_w1807


@pytest.mark.parametrize(
    "fam",
    [
        "bench_hannahanna2_qa_studies_family",
        "bench_ilib2_qa_studies_family",
        "bench_kamrusepa2_qa_studies_family",
        "bench_kumarbi2_qa_studies_family",
        "bench_pirinkir2_qa_studies_family",
        "bench_tesub2_qa_studies_family",
    ],
)
def test_benches_w1807(fam):
    out = getattr(benches_w1807, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
