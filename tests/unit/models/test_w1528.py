import pytest


@pytest.mark.parametrize(
    "name",
    [
        "feldspar_qa_studies",
        "gypsum_qa_studies",
        "calcite_qa_studies",
        "fluorite_qa_studies",
        "olivine_qa_studies",
        "quartz_qa_studies",
    ],
)
def test_w1528_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
