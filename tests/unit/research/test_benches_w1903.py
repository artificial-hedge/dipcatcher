import pytest

from quant_fund.research import benches_w1903


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chert_qa_studies_family",
        "bench_likho_qa_studies_family",
        "bench_polevoy_qa_studies_family",
        "bench_rarog_qa_studies_family",
        "bench_vodyanoy_qa_studies_family",
        "bench_zmey_gorynych_qa_studies_family",
    ],
)
def test_benches_w1903(fam):
    out = getattr(benches_w1903, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
