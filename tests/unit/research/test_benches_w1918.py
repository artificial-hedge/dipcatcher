import pytest

from quant_fund.research import benches_w1918


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bocan_qa_studies_family",
        "bench_dunter_qa_studies_family",
        "bench_fuath_qa_studies_family",
        "bench_redcap_qa_studies_family",
        "bench_seonaidh_qa_studies_family",
        "bench_wraith_qa_studies_family",
    ],
)
def test_benches_w1918(fam):
    out = getattr(benches_w1918, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
