import pytest

from quant_fund.research import benches_w1602


@pytest.mark.parametrize(
    "fam",
    [
        "bench_desman_qa_studies_family",
        "bench_marsupial_mole_qa_studies_family",
        "bench_moles_lite_qa_studies_family",
        "bench_monotreme_qa_studies_family",
        "bench_moonrat_qa_studies_family",
        "bench_sengi_qa_studies_family",
    ],
)
def test_benches_w1602(fam):
    out = getattr(benches_w1602, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
