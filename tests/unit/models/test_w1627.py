import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cave_scorpion_qa_studies",
        "cave_worm_qa_studies",
        "cave_crayfish_qa_studies",
        "cave_springtail_qa_studies",
        "troglofish_qa_studies",
        "stygobite_qa_studies",
    ],
)
def test_w1627_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
