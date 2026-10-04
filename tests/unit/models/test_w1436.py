import pytest


@pytest.mark.parametrize(
    "name",
    [
        "galaxy_qa_studies",
        "nebula_qa_studies",
        "moon_qa_studies",
        "planet_qa_studies",
        "star_qa_studies",
        "comet_qa_studies",
    ],
)
def test_w1436_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
