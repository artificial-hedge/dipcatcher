import pytest


@pytest.mark.parametrize(
    "name",
    [
        "flame_qa_studies",
        "ironic_qa_studies",
        "hate_qa_studies",
        "offensive_qa_studies",
        "politeness_qa_studies",
        "fakeqa_lite_studies",
    ],
)
def test_w1411_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
