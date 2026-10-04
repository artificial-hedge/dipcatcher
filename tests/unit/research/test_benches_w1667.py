import pytest

from quant_fund.research import benches_w1667


@pytest.mark.parametrize(
    "fam",
    [
        "bench_centauride_qa_studies_family",
        "bench_dryad_qa_studies_family",
        "bench_faun_qa_studies_family",
        "bench_hamadryad_qa_studies_family",
        "bench_nereid_qa_studies_family",
        "bench_nymph_qa_studies_family",
    ],
)
def test_benches_w1667(fam):
    out = getattr(benches_w1667, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
