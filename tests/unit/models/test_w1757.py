import pytest


@pytest.mark.parametrize(
    "name",
    [
        "danu_qa_studies",
        "brigid_qa_studies",
        "dagda_qa_studies",
        "manannan_qa_studies",
        "morgen_qa_studies",
        "aisling_qa_studies",
    ],
)
def test_w1757_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
