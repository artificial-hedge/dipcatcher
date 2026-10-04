import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sturgeon_qa_studies",
        "pike_qa_studies",
        "perch_qa_studies",
        "walleye_qa_studies",
        "crappie_qa_studies",
        "bluegill_qa_studies",
    ],
)
def test_w1566_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
