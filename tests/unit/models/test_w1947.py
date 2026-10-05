import pytest


@pytest.mark.parametrize(
    "name",
    [
        "baron_samedi_qa_studies",
        "papa_legba_qa_studies",
        "ezili_dantor_qa_studies",
        "damballa_qa_studies",
        "ogou_feray_qa_studies",
        "simbi_qa_studies",
    ],
)
def test_w1947_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
