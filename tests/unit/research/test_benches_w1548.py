import pytest

from quant_fund.research import benches_w1548


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ani_qa_studies_family",
        "bench_coua_qa_studies_family",
        "bench_guira_qa_studies_family",
        "bench_hoatzin_qa_studies_family",
        "bench_malkoha_qa_studies_family",
        "bench_turaco_qa_studies_family",
    ],
)
def test_benches_w1548(fam):
    out = getattr(benches_w1548, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
