import pytest


@pytest.mark.parametrize(
    "name",
    [
        "fishing_cat_qa_studies",
        "jungle_cat_qa_studies",
        "black_footed_qa_studies",
        "pallas_qa_studies",
        "rusty_spotted_qa_studies",
        "sand_cat_qa_studies",
    ],
)
def test_w1590_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
