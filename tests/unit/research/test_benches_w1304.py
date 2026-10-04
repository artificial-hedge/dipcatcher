import pytest

from quant_fund.research import benches_w1304


@pytest.mark.parametrize(
    "fam",
    [
        "bench_boolq_studies_family",
        "bench_copa_studies_family",
        "bench_hellaswag_studies_family",
        "bench_openbookqa_studies_family",
        "bench_piqa_studies_family",
        "bench_siqa_studies_family",
    ],
)
def test_benches_w1304(fam):
    out = getattr(benches_w1304, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
