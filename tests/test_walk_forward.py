"""
Tests for src/validation/walk_forward.py
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validation.walk_forward import WalkForwardSplitter


class TestWalkForwardSplitterInit:

    def test_invalid_initial_window_raises(self):
        with pytest.raises(ValueError, match="initial_window"):
            WalkForwardSplitter(initial_window=0, horizon=21)

    def test_negative_initial_window_raises(self):
        with pytest.raises(ValueError, match="initial_window"):
            WalkForwardSplitter(initial_window=-1, horizon=21)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            WalkForwardSplitter(initial_window=750, horizon=0)

    def test_negative_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            WalkForwardSplitter(initial_window=750, horizon=-5)


class TestWalkForwardSplitterSplit:

    def test_happy_path_fold_count(self):
        # n=1250, initial_window=750, horizon=21 → (1250-750)//21 = 23 folds
        splitter = WalkForwardSplitter(initial_window=750, horizon=21)
        folds = splitter.split(1250)
        assert len(folds) == 23

    def test_exact_boundary_one_fold(self):
        # n == initial_window + horizon → exactly 1 fold
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        folds = splitter.split(15)
        assert len(folds) == 1

    def test_n_too_small_raises(self):
        splitter = WalkForwardSplitter(initial_window=750, horizon=21)
        with pytest.raises(ValueError, match="Not enough data"):
            splitter.split(770)  # 750 + 21 = 771, so 770 is too small

    def test_n_equals_minimum_minus_one_raises(self):
        splitter = WalkForwardSplitter(initial_window=5, horizon=3)
        with pytest.raises(ValueError):
            splitter.split(7)  # 5 + 3 = 8 needed; 7 is one short

    def test_train_indices_expand_by_horizon(self):
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        folds = splitter.split(30)
        # Fold 0: train [0:10], Fold 1: train [0:15], Fold 2: train [0:20], Fold 3: train [0:25]
        assert len(folds[0][0]) == 10
        assert len(folds[1][0]) == 15
        assert len(folds[2][0]) == 20

    def test_train_indices_start_from_zero_every_fold(self):
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        folds = splitter.split(30)
        for train_idx, _ in folds:
            assert train_idx[0] == 0

    def test_test_windows_are_contiguous_and_non_overlapping(self):
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        folds = splitter.split(30)
        for i in range(1, len(folds)):
            prev_test_end = folds[i - 1][1][-1] + 1
            curr_test_start = folds[i][1][0]
            assert curr_test_start == prev_test_end, (
                f"Fold {i} test starts at {curr_test_start}, expected {prev_test_end}"
            )

    def test_test_window_fixed_at_horizon(self):
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        folds = splitter.split(40)
        for _, test_idx in folds:
            assert len(test_idx) == 5

    def test_first_test_window_starts_at_initial_window(self):
        splitter = WalkForwardSplitter(initial_window=750, horizon=21)
        folds = splitter.split(1250)
        assert folds[0][1][0] == 750

    def test_no_overlap_between_train_and_test(self):
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        folds = splitter.split(30)
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Fold {fold_id} has train/test overlap: {overlap}"


class TestNFolds:

    def test_n_folds_matches_len_split(self):
        splitter = WalkForwardSplitter(initial_window=750, horizon=21)
        assert splitter.n_folds(1250) == len(splitter.split(1250))

    def test_n_folds_too_small_returns_zero(self):
        splitter = WalkForwardSplitter(initial_window=750, horizon=21)
        assert splitter.n_folds(770) == 0

    def test_n_folds_exact_minimum(self):
        splitter = WalkForwardSplitter(initial_window=10, horizon=5)
        assert splitter.n_folds(15) == 1
