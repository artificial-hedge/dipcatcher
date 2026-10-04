import pytest

from quant_fund.research import benches_w1694


@pytest.mark.parametrize(
    "fam",
    [
        "bench_duwende_qa_studies_family",
        "bench_karibusa_qa_studies_family",
        "bench_mambabarang_qa_studies_family",
        "bench_mangkukulam_qa_studies_family",
        "bench_sokoy_qa_studies_family",
        "bench_tiktik_qa_studies_family",
    ],
)
def test_benches_w1694(fam):
    out = getattr(benches_w1694, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
