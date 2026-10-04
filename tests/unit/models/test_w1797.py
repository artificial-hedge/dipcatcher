import pytest


@pytest.mark.parametrize(
    "name",
    [
        "lachesis_qa_studies",
        "atropos_qa_studies",
        "eunomia_qa_studies",
        "peitho_qa_studies",
        "dikaion_qa_studies",
        "metis2_qa_studies",
    ],
)
def test_w1797_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
