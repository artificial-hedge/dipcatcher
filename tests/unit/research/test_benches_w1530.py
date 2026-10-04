import pytest

from quant_fund.research import benches_w1530


@pytest.mark.parametrize(
    "fam",
    [
        "bench_amalgam_qa_studies_family",
        "bench_brass_qa_studies_family",
        "bench_bronze_qa_studies_family",
        "bench_nichrome_qa_studies_family",
        "bench_pewter_qa_studies_family",
        "bench_solder_qa_studies_family",
    ],
)
def test_benches_w1530(fam):
    out = getattr(benches_w1530, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
