from typing import Tuple

import numpy as np
import pandas as pd


class NaiveBaseline:
    """
    1-step-ahead random-walk baseline: predict[t] = actual[t-1].

      pred[0]   = y_train.iloc[-1]   (last known value before the test window)
      pred[i>0] = y_test.iloc[i-1]   (previous test-window actual)

    This matches how financial economists benchmark: tomorrow = today.
    """

    def __init__(self, cfg, features: pd.DataFrame, logger, ticker: str) -> None:
        self.cfg = cfg
        self.features = features
        self.logger = logger
        self.ticker = ticker

    def predict_fold(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions, residuals) both of shape (len(y_test),).

        Raises ValueError if y_train or y_test is empty.
        """
        y_train_vals = np.asarray(y_train, dtype=float).ravel()
        y_test_vals = np.asarray(y_test, dtype=float).ravel()

        if len(y_train_vals) == 0:
            raise ValueError("y_train must be non-empty for NaiveBaseline.predict_fold()")
        if len(y_test_vals) == 0:
            raise ValueError("y_test must be non-empty for NaiveBaseline.predict_fold()")

        n = len(y_test_vals)
        predictions = np.empty(n, dtype=float)
        predictions[0] = y_train_vals[-1]
        if n > 1:
            predictions[1:] = y_test_vals[:-1]

        residuals = y_test_vals - predictions
        return predictions, residuals
