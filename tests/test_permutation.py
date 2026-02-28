"""Tests for the Webb permutation robustness analysis."""

import numpy as np
import pytest

from evidence_synthesis.analysis.permutation_webb2012 import (
    generate_surjective_assignments,
    canonicalize,
    build_design_matrix_from_assignment,
    run_permutation_test,
)


class TestCanonicalize:
    def test_identity(self):
        assert canonicalize((0, 1, 2)) == (0, 1, 2)

    def test_relabels_by_first_appearance(self):
        assert canonicalize((2, 0, 1, 2)) == (0, 1, 2, 0)

    def test_single_group(self):
        assert canonicalize((3, 3, 3)) == (0, 0, 0)

    def test_two_groups(self):
        assert canonicalize((1, 0, 1, 0)) == (0, 1, 0, 1)


class TestSurjectiveAssignments:
    def test_small_case_count(self):
        """S(3,2) = 3, times 2! = 6 surjective functions, 3 canonical."""
        assignments = list(generate_surjective_assignments(3, 2))
        assert len(assignments) == 6  # all surjective, before canonicalization
        canonical = set(canonicalize(a) for a in assignments)
        assert len(canonical) == 3  # S(3,2) = 3

    def test_s_10_3_count(self):
        """S(10,3) = 9,330 unique partitions (canonical)."""
        canonical = set()
        for assignment in generate_surjective_assignments(10, 3):
            canonical.add(canonicalize(assignment))
        assert len(canonical) == 9330

    def test_all_groups_nonempty(self):
        """Every surjective assignment must use all k groups."""
        for assignment in generate_surjective_assignments(5, 3):
            assert len(set(assignment)) == 3


class TestBuildDesignMatrix:
    def test_shape(self):
        assignment = (0, 0, 1, 1, 2)
        X = build_design_matrix_from_assignment(assignment)
        assert X.shape == (5, 3)  # intercept + 2 dummies

    def test_reference_group_zero_dummies(self):
        assignment = (0, 1, 2)
        X = build_design_matrix_from_assignment(assignment)
        # Item 0 is group 0 (reference): dummies should be [1, 0, 0]
        np.testing.assert_array_equal(X[0], [1.0, 0.0, 0.0])

    def test_intercept_all_ones(self):
        assignment = (0, 1, 2, 0, 1)
        X = build_design_matrix_from_assignment(assignment)
        np.testing.assert_array_equal(X[:, 0], np.ones(5))


class TestPermutationTest:
    def test_known_best_grouping(self):
        """When data has a clear 3-group structure, the true grouping should
        achieve among the lowest tau-squared values."""
        # Three distinct groups
        y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        v = np.full(9, 0.001)
        # The 'true' grouping: items 0-2 = A, 3-5 = B, 6-8 = C
        true_assignment = (0, 0, 0, 1, 1, 1, 2, 2, 2)
        from evidence_synthesis.analysis.meta_pipeline import fit_reml
        X_true = build_design_matrix_from_assignment(true_assignment)
        res_true = fit_reml(y, v, X_true)

        perm_tau2, p_value = run_permutation_test(y, v, res_true.tau2, n_groups=3)
        # True grouping should be in the best 5%
        assert p_value <= 0.05, f"p={p_value}: true grouping should be among the best"

    def test_p_value_in_range(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        v = np.full(5, 0.01)
        perm_tau2, p_value = run_permutation_test(y, v, 0.01, n_groups=3)
        assert 0.0 <= p_value <= 1.0

    def test_random_grouping_nonsignificant(self):
        """A random grouping of homogeneous data shouldn't be special."""
        rng = np.random.default_rng(42)
        y = rng.normal(0.3, 0.02, size=8)
        v = np.full(8, 0.01)
        # Use a random assignment
        random_assignment = (0, 1, 2, 0, 1, 2, 0, 1)
        from evidence_synthesis.analysis.meta_pipeline import fit_reml
        X = build_design_matrix_from_assignment(random_assignment)
        res = fit_reml(y, v, X)
        _, p_value = run_permutation_test(y, v, res.tau2, n_groups=3)
        # For homogeneous data, random grouping should not be significant
        assert p_value > 0.01
