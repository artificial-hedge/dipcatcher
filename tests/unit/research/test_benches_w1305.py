import pytest

from quant_fund.research import benches_w1305


@pytest.mark.parametrize(
    "fam",
    [
        "bench_coqa_studies_family",
        "bench_drop_studies_family",
        "bench_hotpotqa_studies_family",
        "bench_nq_studies_family",
        "bench_squad_studies_family",
        "bench_triviaqa_studies_family",
    ],
)
def test_benches_w1305(fam):
    out = getattr(benches_w1305, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
