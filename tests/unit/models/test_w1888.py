import pytest


@pytest.mark.parametrize(
    "name",
    [
        "voluspa_qa_studies",
        "grimnismal_qa_studies",
        "havamal_qa_studies",
        "vafthrudnir_qa_studies",
        "skirnismal_qa_studies",
        "lokasenna_qa_studies",
    ],
)
def test_w1888_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
