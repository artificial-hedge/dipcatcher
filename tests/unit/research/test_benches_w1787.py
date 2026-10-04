import pytest

from quant_fund.research import benches_w1787


@pytest.mark.parametrize(
    "fam",
    [
        "bench_eir_qa_studies_family",
        "bench_heimdal_qa_studies_family",
        "bench_norns_qa_studies_family",
        "bench_odin_qa_studies_family",
        "bench_thor_qa_studies_family",
        "bench_valkyrie_qa_studies_family",
    ],
)
def test_benches_w1787(fam):
    out = getattr(benches_w1787, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
