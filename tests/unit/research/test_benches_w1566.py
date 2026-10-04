import pytest

from quant_fund.research import benches_w1566


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bluegill_qa_studies_family",
        "bench_crappie_qa_studies_family",
        "bench_perch_qa_studies_family",
        "bench_pike_qa_studies_family",
        "bench_sturgeon_qa_studies_family",
        "bench_walleye_qa_studies_family",
    ],
)
def test_benches_w1566(fam):
    out = getattr(benches_w1566, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
