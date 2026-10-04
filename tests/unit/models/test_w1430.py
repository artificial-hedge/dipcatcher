import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bird_qa_studies",
        "fish_qa_studies",
        "ecosystem_qa_studies",
        "habitat_qa_studies",
        "insect_qa_studies",
        "animal_qa_studies",
    ],
)
def test_w1430_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
