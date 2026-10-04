import pytest

from quant_fund.research import benches_w1331


@pytest.mark.parametrize(
    "fam",
    [
        "bench_codescope_studies_family",
        "bench_concode_eval_studies_family",
        "bench_crosscodeeval_studies_family",
        "bench_mer_bench_studies_family",
        "bench_project_eval_studies_family",
        "bench_swe_bench_verified_studies_family",
    ],
)
def test_benches_w1331(fam):
    out = getattr(benches_w1331, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
