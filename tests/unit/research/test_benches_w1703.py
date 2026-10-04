import pytest

from quant_fund.research import benches_w1703


@pytest.mark.parametrize(
    "fam",
    [
        "bench_dumuzi_qa_studies_family",
        "bench_inanna_qa_studies_family",
        "bench_marduk_qa_studies_family",
        "bench_namtar_qa_studies_family",
        "bench_nergal_qa_studies_family",
        "bench_ninhursag_qa_studies_family",
    ],
)
def test_benches_w1703(fam):
    out = getattr(benches_w1703, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
