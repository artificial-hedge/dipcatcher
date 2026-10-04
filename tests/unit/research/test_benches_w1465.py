import pytest

from quant_fund.research import benches_w1465


@pytest.mark.parametrize(
    "fam",
    [
        "bench_citadel_qa_studies_family",
        "bench_forge_qa_studies_family",
        "bench_grotto_qa_studies_family",
        "bench_lighthouse_qa_studies_family",
        "bench_quarry_qa_studies_family",
        "bench_vault_qa_studies_family",
    ],
)
def test_benches_w1465(fam):
    out = getattr(benches_w1465, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
