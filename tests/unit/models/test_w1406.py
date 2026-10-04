import pytest


@pytest.mark.parametrize(
    "name",
    [
        "syndq_lite_studies",
        "time_qa_studies",
        "teas_qa_studies",
        "timedial_qa_studies",
        "timetravel_lite_studies",
        "menat_qa_studies",
    ],
)
def test_w1406_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
