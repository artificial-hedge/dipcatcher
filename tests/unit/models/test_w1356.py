import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mmlu_lite_studies",
        "bigbench_lite_studies",
        "triviaqa_lite_studies",
        "natural_qa_studies",
        "pop_qa_studies",
        "entity_qa_studies",
    ],
)
def test_w1356_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
