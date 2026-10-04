import pytest

from quant_fund.research import benches_w1306


@pytest.mark.parametrize(
    "fam",
    [
        "bench_glue_studies_family",
        "bench_mnli_studies_family",
        "bench_qnli_studies_family",
        "bench_rte_studies_family",
        "bench_super_glue_studies_family",
        "bench_wnli_studies_family",
    ],
)
def test_benches_w1306(fam):
    out = getattr(benches_w1306, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
