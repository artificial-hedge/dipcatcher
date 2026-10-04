import pytest


@pytest.mark.parametrize(
    "name",
    [
        "night_heron_qa_studies",
        "grey_heron_qa_studies",
        "tiger_heron_qa_studies",
        "green_heron_qa_studies",
        "goliath_heron_qa_studies",
        "purple_heron_qa_studies",
    ],
)
def test_w1539_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
