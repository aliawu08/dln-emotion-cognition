"""Tests for the Greenwald (2009) domain-level permutation test."""

import numpy as np
import pytest

from evidence_synthesis.analysis.permutation_greenwald2009_domain import (
    generate_surjective_assignments,
    canonicalize,
    build_design_matrix_from_domain_assignment,
    run_domain_permutation_test,
)


class TestSurjectiveEnumeration:
    def test_s_9_4_count(self):
        """S(9,4) = 7,770 unique canonical partitions."""
        canonical = set()
        for assignment in generate_surjective_assignments(9, 4):
            canonical.add(canonicalize(assignment))
        assert len(canonical) == 7770

    def test_s_4_2_count(self):
        """S(4,2) = 7 unique canonical partitions."""
        canonical = set()
        for assignment in generate_surjective_assignments(4, 2):
            canonical.add(canonicalize(assignment))
        assert len(canonical) == 7

    def test_all_groups_nonempty(self):
        """Every surjective assignment must use all k groups."""
        for assignment in generate_surjective_assignments(5, 3):
            assert len(set(assignment)) == 3


class TestCanonicalize:
    def test_identity(self):
        assert canonicalize((0, 1, 2, 3)) == (0, 1, 2, 3)

    def test_relabels(self):
        assert canonicalize((3, 1, 0, 2)) == (0, 1, 2, 3)

    def test_single_group(self):
        assert canonicalize((5, 5, 5)) == (0, 0, 0)


class TestDesignMatrix:
    def test_shape(self):
        domains = ["A", "B", "C"]
        assignment = (0, 1, 2)
        studies = np.array(["A", "A", "B", "C", "C"])
        X = build_design_matrix_from_domain_assignment(domains, assignment, studies)
        assert X.shape == (5, 3)

    def test_intercept_all_ones(self):
        domains = ["A", "B"]
        assignment = (0, 1)
        studies = np.array(["A", "B", "A"])
        X = build_design_matrix_from_domain_assignment(domains, assignment, studies)
        np.testing.assert_array_equal(X[:, 0], np.ones(3))

    def test_grouping_correct(self):
        domains = ["A", "B", "C"]
        assignment = (0, 0, 1)  # A,B -> group 0; C -> group 1
        studies = np.array(["A", "B", "C"])
        X = build_design_matrix_from_domain_assignment(domains, assignment, studies)
        # A -> group 0 (reference): [1, 0]
        np.testing.assert_array_equal(X[0], [1.0, 0.0])
        # B -> group 0 (reference): [1, 0]
        np.testing.assert_array_equal(X[1], [1.0, 0.0])
        # C -> group 1: [1, 1]
        np.testing.assert_array_equal(X[2], [1.0, 1.0])


class TestDomainPermutation:
    def test_p_value_in_range(self):
        """P-value must be between 0 and 1."""
        rng = np.random.default_rng(42)
        # 4 domains, each with 5 studies
        domains = np.repeat(["A", "B", "C", "D"], 5)
        y = rng.normal(0.3, 0.1, size=20)
        v = np.full(20, 0.01)
        _, p_value, n_parts = run_domain_permutation_test(
            y, v, domains, dln_tau2=0.01, n_groups=2
        )
        assert 0.0 <= p_value <= 1.0
        # S(4,2) = 7
        assert n_parts == 7

    def test_clear_structure_significant(self):
        """When domains have distinct true means, the correct grouping should
        achieve a low tau-squared."""
        # 3 domains with clearly separated means
        domains = np.array(["lo"] * 10 + ["mid"] * 10 + ["hi"] * 10)
        y = np.concatenate([
            np.full(10, 0.0),
            np.full(10, 0.5),
            np.full(10, 1.0),
        ])
        v = np.full(30, 0.001)

        # The "true" grouping: each domain in its own group
        from evidence_synthesis.analysis.meta_pipeline import fit_reml
        # Build the correct design matrix
        X_true = np.column_stack([
            np.ones(30),
            (domains == "lo").astype(float),
            (domains == "mid").astype(float),
        ])
        res_true = fit_reml(y, v, X_true)

        _, p_value, n_parts = run_domain_permutation_test(
            y, v, domains, dln_tau2=res_true.tau2, n_groups=3
        )
        # S(3,3) = 1 unique partition — only one way to put 3 items into 3 groups
        assert n_parts == 1
        assert p_value == 1.0  # the only partition IS the correct one
