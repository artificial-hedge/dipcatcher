import pytest


@pytest.mark.parametrize(
    "name",
    [
        "red_deer_qa_studies",
        "mule_qa_studies",
        "kouprey_qa_studies",
        "wapiti_qa_studies",
        "pere_david_qa_studies",
        "hog_deer_qa_studies",
    ],
)
def test_w1614_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
