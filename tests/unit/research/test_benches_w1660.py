import pytest

from quant_fund.research import benches_w1660


@pytest.mark.parametrize(
    "fam",
    [
        "bench_centaur_qa_studies_family",
        "bench_cyclops_qa_studies_family",
        "bench_griffin_qa_studies_family",
        "bench_hydra_qa_studies_family",
        "bench_medusa_qa_studies_family",
        "bench_sphinx_qa_studies_family",
    ],
)
def test_benches_w1660(fam):
    out = getattr(benches_w1660, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
