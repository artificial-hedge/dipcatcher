import pytest

from quant_fund.research import benches_w1784


@pytest.mark.parametrize(
    "fam",
    [
        "bench_ares_qa_studies_family",
        "bench_hades_qa_studies_family",
        "bench_hephaestus_qa_studies_family",
        "bench_hestia_qa_studies_family",
        "bench_poseidon_qa_studies_family",
        "bench_zeus_qa_studies_family",
    ],
)
def test_benches_w1784(fam):
    out = getattr(benches_w1784, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
