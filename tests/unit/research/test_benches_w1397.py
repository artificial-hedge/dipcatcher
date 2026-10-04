import pytest

from quant_fund.research import benches_w1397


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coma_qa_studies_family",
        "bench_gaia_lite_studies_family",
        "bench_simple_qa_studies_family",
        "bench_sqa_lite_studies_family",
        "bench_tqa_lite_studies_family",
        "bench_tydiqa_lite_studies_family",
    ],
)
def test_benches_w1397(fam):
    out = getattr(benches_w1397, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
