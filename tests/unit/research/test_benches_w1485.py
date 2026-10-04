import pytest

from quant_fund.research import benches_w1485


@pytest.mark.parametrize(
    "fam",
    [
        "bench_avocet_qa_studies_family",
        "bench_egret_qa_studies_family",
        "bench_heron_qa_studies_family",
        "bench_plover_qa_studies_family",
        "bench_sandpiper_qa_studies_family",
        "bench_tern_qa_studies_family",
    ],
)
def test_benches_w1485(fam):
    out = getattr(benches_w1485, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
