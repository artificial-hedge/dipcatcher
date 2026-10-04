import pytest

from quant_fund.research import benches_w1913


@pytest.mark.parametrize(
    "fam",
    [
        "bench_berstuk_qa_studies_family",
        "bench_chuma_qa_studies_family",
        "bench_koshmar_qa_studies_family",
        "bench_navka_qa_studies_family",
        "bench_perekus_qa_studies_family",
        "bench_rugievit_qa_studies_family",
    ],
)
def test_benches_w1913(fam):
    out = getattr(benches_w1913, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
