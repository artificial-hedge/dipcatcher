import pytest


@pytest.mark.parametrize(
    "name",
    [
        "leto_qa_studies",
        "eileithyia_qa_studies",
        "nemesis_qa_studies",
        "tyche_qa_studies",
        "iris_qa_studies",
        "nike_qa_studies",
    ],
)
def test_w1764_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
