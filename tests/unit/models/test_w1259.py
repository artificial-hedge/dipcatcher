import pytest


@pytest.mark.parametrize("name", ['qsar_studies', 'docking_studies', 'admet_studies', 'lead_optimization_studies', 'virtual_screening_studies', 'de_novo_design_studies'])
def test_w1259_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
