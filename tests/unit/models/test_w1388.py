import pytest


@pytest.mark.parametrize(
    "name",
    [
        "corpus_qa_studies",
        "freshqa_studies",
        "domain_rag_studies",
        "ragas_lite_studies",
        "rgb_eval_studies",
        "crag_bench_studies",
    ],
)
def test_w1388_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
