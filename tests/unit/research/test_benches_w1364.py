import pytest

from quant_fund.research import benches_w1364


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anli_lite_studies_family",
        "bench_mnli_lite_studies_family",
        "bench_mrpc_lite_studies_family",
        "bench_paws_lite_studies_family",
        "bench_quora_dup_studies_family",
        "bench_rte_lite_studies_family",
    ],
)
def test_benches_w1364(fam):
    out = getattr(benches_w1364, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
