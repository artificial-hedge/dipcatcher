import pytest


@pytest.mark.parametrize(
    "name",
    [
        "white_footed_qa_studies",
        "russet_qa_studies",
        "anosy_qa_studies",
        "red_bellied_qa_studies",
        "daraina_qa_studies",
        "amber_mountain_qa_studies",
    ],
)
def test_w1617_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
