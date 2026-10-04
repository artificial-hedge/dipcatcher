import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cave_fish_qa_studies",
        "troglobite_qa_studies",
        "mudpuppy_qa_studies",
        "olm_qa_studies",
        "cave_beetle_qa_studies",
        "cave_cricket_qa_studies",
    ],
)
def test_w1621_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
