import pytest

from quant_fund.research import benches_w1358


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adver_qa_studies_family",
        "bench_coqa_lite_studies_family",
        "bench_drop_lite_studies_family",
        "bench_duo_rc_studies_family",
        "bench_quac_lite_studies_family",
        "bench_trivia_web_studies_family",
    ],
)
def test_benches_w1358(fam):
    out = getattr(benches_w1358, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
