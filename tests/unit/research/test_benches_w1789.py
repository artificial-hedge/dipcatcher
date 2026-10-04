import pytest

from quant_fund.research import benches_w1789


@pytest.mark.parametrize(
    "fam",
    [
        "bench_eris_qa_studies_family",
        "bench_ganymede_qa_studies_family",
        "bench_hebe_qa_studies_family",
        "bench_hermes_qa_studies_family",
        "bench_momus_qa_studies_family",
        "bench_oneiros_qa_studies_family",
    ],
)
def test_benches_w1789(fam):
    out = getattr(benches_w1789, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
