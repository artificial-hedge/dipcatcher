import pytest

from quant_fund.research import benches_w1463


@pytest.mark.parametrize(
    "fam",
    [
        "bench_acorn_qa_studies_family",
        "bench_blossom_qa_studies_family",
        "bench_canopy_qa_studies_family",
        "bench_firefly_qa_studies_family",
        "bench_sprout_qa_studies_family",
        "bench_truffle_qa_studies_family",
    ],
)
def test_benches_w1463(fam):
    out = getattr(benches_w1463, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
