import pytest

from quant_fund.research import benches_w1671


@pytest.mark.parametrize(
    "fam",
    [
        "bench_drakk_qa_studies_family",
        "bench_grimr_qa_studies_family",
        "bench_hildr_qa_studies_family",
        "bench_mare_qa_studies_family",
        "bench_nisse_qa_studies_family",
        "bench_sigrun_qa_studies_family",
    ],
)
def test_benches_w1671(fam):
    out = getattr(benches_w1671, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
