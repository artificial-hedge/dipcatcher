import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ammit_qa_studies",
        "sphairo_qa_studies",
        "apophis_qa_studies",
        "akhekh_qa_studies",
        "sekhmet_qa_studies",
        "bes_qa_studies",
    ],
)
def test_w1647_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
