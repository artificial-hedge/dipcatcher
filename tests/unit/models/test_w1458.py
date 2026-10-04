import pytest


@pytest.mark.parametrize(
    "name",
    [
        "butte_qa_studies",
        "caravan_qa_studies",
        "camel_qa_studies",
        "mirage_qa_studies",
        "oasis_qa_studies",
        "arroyo_qa_studies",
    ],
)
def test_w1458_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
