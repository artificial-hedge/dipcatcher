import pytest

from quant_fund.research import benches_w1901


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boitata_qa_studies_family",
        "bench_boto_qa_studies_family",
        "bench_curupira_qa_studies_family",
        "bench_iara_qa_studies_family",
        "bench_mapinguari_qa_studies_family",
        "bench_saci_qa_studies_family",
    ],
)
def test_benches_w1901(fam):
    out = getattr(benches_w1901, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
