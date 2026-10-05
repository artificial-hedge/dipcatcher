import pytest

from quant_fund.research import benches_w1958


@pytest.mark.parametrize(
    "fam",
    [
        "bench_bathin_qa_studies_family",
        "bench_buer_qa_studies_family",
        "bench_marax_qa_studies_family",
        "bench_marbas_qa_studies_family",
        "bench_purson_qa_studies_family",
        "bench_sallos_qa_studies_family",
    ],
)
def test_benches_w1958(fam):
    out = getattr(benches_w1958, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
