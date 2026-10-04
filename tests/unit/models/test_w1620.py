import pytest


@pytest.mark.parametrize(
    "name",
    [
        "sea_krait_qa_studies",
        "gaboon_qa_studies",
        "inland_taipan_qa_studies",
        "death_adder_qa_studies",
        "boomslang_qa_studies",
        "saw_scaled_qa_studies",
    ],
)
def test_w1620_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
