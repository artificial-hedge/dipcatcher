import pytest

from quant_fund.research import benches_w1645


@pytest.mark.parametrize(
    "fam",
    [
        "bench_argus_qa_studies_family",
        "bench_cerberus_2_qa_studies_family",
        "bench_nemean_qa_studies_family",
        "bench_orthrus_qa_studies_family",
        "bench_pegasus_2_qa_studies_family",
        "bench_typhon_qa_studies_family",
    ],
)
def test_benches_w1645(fam):
    out = getattr(benches_w1645, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
