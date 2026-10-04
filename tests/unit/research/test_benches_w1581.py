import pytest

from quant_fund.research import benches_w1581


@pytest.mark.parametrize(
    "fam",
    [
        "bench_elephant_seal_qa_studies_family",
        "bench_fur_seal_qa_studies_family",
        "bench_harp_seal_qa_studies_family",
        "bench_leopard_seal_qa_studies_family",
        "bench_monk_seal_qa_studies_family",
        "bench_weddell_qa_studies_family",
    ],
)
def test_benches_w1581(fam):
    out = getattr(benches_w1581, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
