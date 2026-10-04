import pytest

from quant_fund.research import benches_w1575


@pytest.mark.parametrize(
    "fam",
    [
        "bench_chinchilla_qa_studies_family",
        "bench_degu_qa_studies_family",
        "bench_gerbil_qa_studies_family",
        "bench_hamster_qa_studies_family",
        "bench_lemming_qa_studies_family",
        "bench_vole_qa_studies_family",
    ],
)
def test_benches_w1575(fam):
    out = getattr(benches_w1575, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
