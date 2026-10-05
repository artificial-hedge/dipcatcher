import pytest


@pytest.mark.parametrize(
    "name",
    [
        "polevoy_qa_studies",
        "rarog_qa_studies",
        "likho_qa_studies",
        "vodyanoy_qa_studies",
        "chert_qa_studies",
        "zmey_gorynych_qa_studies",
    ],
)
def test_w1903_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
