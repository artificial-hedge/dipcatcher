import pytest

from quant_fund.research import benches_w1803


@pytest.mark.parametrize(
    "fam",
    [
        "bench_andrasta2_qa_studies_family",
        "bench_borvo2_qa_studies_family",
        "bench_epona2_qa_studies_family",
        "bench_etercuni2_qa_studies_family",
        "bench_maponos2_qa_studies_family",
        "bench_rosmerta2_qa_studies_family",
    ],
)
def test_benches_w1803(fam):
    out = getattr(benches_w1803, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
