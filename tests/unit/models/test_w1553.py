import pytest


@pytest.mark.parametrize(
    "name",
    [
        "leopard_frog_qa_studies",
        "dart_frog_qa_studies",
        "wood_frog_qa_studies",
        "spring_peeper_qa_studies",
        "horned_frog_qa_studies",
        "treefrog_qa_studies",
    ],
)
def test_w1553_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
