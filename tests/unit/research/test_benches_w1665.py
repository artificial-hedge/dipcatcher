import pytest

from quant_fund.research import benches_w1665


@pytest.mark.parametrize(
    "fam",
    [
        "bench_einherjar_qa_studies_family",
        "bench_hati_qa_studies_family",
        "bench_lindworm_qa_studies_family",
        "bench_skoll_qa_studies_family",
        "bench_vargbroder_qa_studies_family",
        "bench_vedrfolnir_qa_studies_family",
    ],
)
def test_benches_w1665(fam):
    out = getattr(benches_w1665, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
