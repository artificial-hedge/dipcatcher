"""Focused contracts for the SYNTHETIC Telepinu study."""

import unittest
from itertools import product

from quant_fund.models.telepinu_qa_studies import (
    bench_telepinu_qa_studies,
    telepinu_qa_studies_aux,
    telepinu_qa_studies_ok,
)


class TelepinuQAStudiesTests(unittest.TestCase):
    def test_both_checks_are_required(self) -> None:
        for fit_ok, sample_ok in product((False, True), repeat=2):
            with self.subTest(fit_ok=fit_ok, sample_ok=sample_ok):
                self.assertIs(
                    telepinu_qa_studies_ok(fit_ok, sample_ok),
                    fit_ok and sample_ok,
                )

    def test_aux_preserves_boolean(self) -> None:
        for aux in (False, True):
            with self.subTest(aux=aux):
                self.assertIs(telepinu_qa_studies_aux(aux), aux)

    def test_benchmark_preserves_synthetic_contract(self) -> None:
        expected = {"synthetic_telepinu_qa_studies": 1.0}
        self.assertEqual(bench_telepinu_qa_studies(), expected)
        for seed in (-1, 0, 1, 42):
            with self.subTest(seed=seed):
                self.assertEqual(bench_telepinu_qa_studies(seed=seed), expected)
