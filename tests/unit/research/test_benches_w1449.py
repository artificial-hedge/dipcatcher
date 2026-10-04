import pytest

from quant_fund.research import benches_w1449


@pytest.mark.parametrize(
    "fam",
    [
        "bench_beluga_qa_studies_family",
        "bench_manatee_qa_studies_family",
        "bench_narwhal_qa_studies_family",
        "bench_orca_qa_studies_family",
        "bench_otter_qa_studies_family",
        "bench_walrus_qa_studies_family",
    ],
)
def test_benches_w1449(fam):
    out = getattr(benches_w1449, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
