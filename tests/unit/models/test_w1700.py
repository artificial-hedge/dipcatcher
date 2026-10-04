import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dagda_qa_studies",
        "morrigan_qa_studies",
        "lugh_qa_studies",
        "brigid_qa_studies",
        "manannan_qa_studies",
        "danu_qa_studies",
    ],
)
def test_w1700_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
