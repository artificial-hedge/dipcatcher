import pytest

from quant_fund.research import benches_w1303


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bbh_studies_family",
        "bench_gsm8k_studies_family",
        "bench_humaneval_studies_family",
        "bench_ifeval_studies_family",
        "bench_mmlu_studies_family",
        "bench_mt_bench_studies_family",
    ],
)
def test_benches_w1303(fam):
    out = getattr(benches_w1303, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
