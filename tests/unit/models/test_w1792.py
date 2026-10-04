import pytest


@pytest.mark.parametrize(
    "name",
    [
        "urukagina_qa_studies",
        "ninsun_qa_studies",
        "lugulbanda_qa_studies",
        "enmerkar2_qa_studies",
        "agga_qa_studies",
        "babbar_qa_studies",
    ],
)
def test_w1792_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
