from typing import List, Tuple

import numpy as np


class WalkForwardSplitter:
    """
    Expanding-window walk-forward splitter for time-series evaluation.

    NOTE: `horizon` here is the size of each test fold in days, NOT the
    forecast horizon. Models produce rolling 1-step-ahead daily forecasts
    within each fold (see src/models/{sarimax,varmax,lstm,...}_model.py
    and the INVARIANT comment in src/main.py near y_series/X_df assembly).

    Training window grows by `horizon` each fold; test window is fixed at `horizon`.

    With n=1250, initial_window=750, horizon=21:
      - 23 complete folds  (500 // 21 = 23)
      - Fold 1:  train [0:750],  test [750:771]
      - Fold 23: train [0:1232], test [1232:1253]
      - 17 observations at the tail are unused (500 - 23*21 = 17)
    """

    def __init__(self, initial_window: int = 750, horizon: int = 21) -> None:
        if initial_window <= 0:
            raise ValueError(f"initial_window must be > 0, got {initial_window}")
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        self.initial_window = initial_window
        self.horizon = horizon

    def split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (train_indices, test_indices) integer-position arrays.

        Raises ValueError if n is too small for even one fold.
        """
        if n < self.initial_window + self.horizon:
            raise ValueError(
                f"Not enough data: n={n} < initial_window={self.initial_window} "
                f"+ horizon={self.horizon} (need at least {self.initial_window + self.horizon})"
            )
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        train_end = self.initial_window
        while train_end + self.horizon <= n:
            splits.append((
                np.arange(0, train_end),
                np.arange(train_end, train_end + self.horizon),
            ))
            train_end += self.horizon
        return splits

    def n_folds(self, n: int) -> int:
        """Returns the number of complete folds for a dataset of length n."""
        if n < self.initial_window + self.horizon:
            return 0
        return (n - self.initial_window) // self.horizon
