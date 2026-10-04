import pytest

from quant_fund.research import benches_w1655


@pytest.mark.parametrize(
    "fam",
    [
        "bench_basiliskcock_qa_studies_family",
        "bench_calygreyhound_qa_studies_family",
        "bench_cocatrix_qa_studies_family",
        "bench_gryps_qa_studies_family",
        "bench_mantygre_qa_studies_family",
        "bench_opinicus_qa_studies_family",
    ],
)
def test_benches_w1655(fam):
    out = getattr(benches_w1655, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
