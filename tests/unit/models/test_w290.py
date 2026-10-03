"""Unit tests for wave-290 chem-informatics canon modules."""

from quant_fund.models.mol_descriptors import h_donors, mol_weight, rotatable
from quant_fund.models.morgan_fp import _bfs_env, morgan
from quant_fund.models.ring_detect import cyclomatic
from quant_fund.models.smiles_parse import parse
from quant_fund.models.substruct import match
from quant_fund.models.tanimoto import tanimoto


def test_parse_benzene():
    a, b = parse("c1ccccc1")
    assert len(a) == 6 and len(b) == 6


def test_parse_branch():
    a, b = parse("CC(=O)O")
    assert len(a) == 4 and b[1][2] == 2


def test_morgan_identical():
    assert morgan(*parse("CCO")) == morgan(*parse("CCO"))


def test_bfs_env():
    a, b = parse("CCO")
    assert _bfs_env(a, b, 0, 1) == frozenset({"C"})


def test_tanimoto_half():
    assert abs(tanimoto(frozenset({1, 2, 3}), frozenset({2, 3, 4})) - 0.5) < 1e-12


def test_mol_weight():
    a, _ = parse("CCO")
    assert abs(mol_weight(a) - 40.021) < 0.01 and h_donors(a) == 1


def test_rotatable():
    a, b = parse("CCCC")
    assert rotatable(a, b) == 1


def test_substruct_benzene():
    assert match(*parse("c1ccccc1"), *parse("Cc1ccccc1"))
    assert not match(*parse("c1ccccc1"), *parse("CCO"))


def test_cyclomatic():
    a, b = parse("c1ccccc1")
    assert cyclomatic(b, len(a)) == 1
