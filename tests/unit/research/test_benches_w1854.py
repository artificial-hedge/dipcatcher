import pytest

from quant_fund.research import benches_w1854


@pytest.mark.parametrize(
    "fam",
    [
        "bench_astarte_punic_qa_studies_family",
        "bench_baal_hammon_qa_studies_family",
        "bench_mekal_qa_studies_family",
        "bench_sid_qa_studies_family",
        "bench_tanit_punic_qa_studies_family",
        "bench_yamm_qa_studies_family",
    ],
)
def test_benches_w1854(fam):
    out = getattr(benches_w1854, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
