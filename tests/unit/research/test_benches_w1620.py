import pytest

from quant_fund.research import benches_w1620


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boomslang_qa_studies_family",
        "bench_death_adder_qa_studies_family",
        "bench_gaboon_qa_studies_family",
        "bench_inland_taipan_qa_studies_family",
        "bench_saw_scaled_qa_studies_family",
        "bench_sea_krait_qa_studies_family",
    ],
)
def test_benches_w1620(fam):
    out = getattr(benches_w1620, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
