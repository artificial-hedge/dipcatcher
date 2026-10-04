import pytest


@pytest.mark.parametrize(
    "name",
    [
        "blossom_qa_studies",
        "firefly_qa_studies",
        "canopy_qa_studies",
        "sprout_qa_studies",
        "truffle_qa_studies",
        "acorn_qa_studies",
    ],
)
def test_w1463_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
