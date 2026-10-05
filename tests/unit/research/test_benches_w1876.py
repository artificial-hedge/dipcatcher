import pytest

from quant_fund.research import benches_w1876


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abdir_qa_studies_family",
        "bench_baal_magon_qa_studies_family",
        "bench_melkob_qa_studies_family",
        "bench_safun_hu_qa_studies_family",
        "bench_shadash_qa_studies_family",
        "bench_sinn_bedri_qa_studies_family",
    ],
)
def test_benches_w1876(fam):
    out = getattr(benches_w1876, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
