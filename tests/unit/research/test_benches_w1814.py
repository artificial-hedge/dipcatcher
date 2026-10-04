import pytest

from quant_fund.research import benches_w1814


@pytest.mark.parametrize(
    "fam",
    [
        "bench_enki2_qa_studies_family",
        "bench_marduk2_qa_studies_family",
        "bench_nanna2_qa_studies_family",
        "bench_ninhursag2_qa_studies_family",
        "bench_tiamat2_qa_studies_family",
        "bench_utu2_qa_studies_family",
    ],
)
def test_benches_w1814(fam):
    out = getattr(benches_w1814, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
