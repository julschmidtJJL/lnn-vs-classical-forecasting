import time
import uuid
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.feature_policy import apply_target_policy


def _make_scaler(kind: str):
    if kind == "zscore":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler kind: {kind!r}. Use 'zscore' or 'minmax'.")
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader



class LSTMModel:

    """
    Thin wrapper around PyTorch LSTM model to fit and produce both in-sample and out-of-sample forecasts.
    """

    # ─── HELPER CLASS ─────────────────────────────────────────────────────────────────────────────────────────────────

    class TorchLSTM(nn.Module):

        """
        Simple LSTM -> Dense head (returns single value).
        """

        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):

            """

            :param n_features:
            :param hidden_size:
            :param num_layers:
            :param dropout:
            """

            super().__init__()

            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                                dropout=dropout if num_layers > 1 else 0.0, batch_first=True)

            self.head = nn.Linear(hidden_size, 1)



        def forward(self, x):

            """

            :param x:
            :return:
            """

            # x: (batch, seq_len, n_features)
            out, _ = self.lstm(x)            # out: (batch, seq_len, hidden)
            last = out[:, -1, :]             # take last time-step output

            return self.head(last).squeeze(-1)  # (batch,)



    def __init__(self, cfg: Dict, features: pd.DataFrame, logger, ticker: str):

        """
        Constructor / Initialising PyTorch VARMA(X) model.

        :param cfg: Your config.yaml file containing all settings.
        :param features: The features dataset we are forecasting on.
        :param logger: SupabaseLogger instance for loggers this models results properly.
        :param ticker: The ticker of the stock currently being analysed.
        """

        self.cfg = cfg
        self.features = features.copy()
        self.logger = logger
        self.ticker = ticker

        # model artifacts
        self.model: Optional[nn.Module] = None
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None

        self.runtime_seconds: Optional[float] = None
        self.last_seq_len: Optional[int] = None
        self.last_params: Optional[dict] = None

        # device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def get_cfg(self) -> dict:

        """
        Returns LSTM hyperparameters with sensible defaults.
        """

        base = self.cfg.get("models", {}).get("LSTM", {})

        return {
            "hidden_size": int(base.get("hidden_size", 64)),
            "num_layers": int(base.get("num_layers", 2)),
            "dropout": float(base.get("dropout", 0.0)),
            "lr": float(base.get("lr", 1e-3)),
            "batch_size": int(base.get("batch_size", 32)),
            "epochs": int(base.get("epochs", 10)),
            "seq_len": int(base.get("seq_len", 20)),
            "forecast_steps": int(base.get("forecast_steps", 10)),
        }



    def sanitize_dataframe(self, df: pd.DataFrame, name: str = "") -> pd.DataFrame:

        """
        Drop or repair invalid columns before model training.

        :param df: The pandas Dataframe being sanitized before fitting the model.
        :param name: The name of the dataset being sanitized.
        """

        n_before = df.shape[1]

        # Remove columns that are entirely NaN or inf
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1, how="all")

        # Optionally fill remaining NaNs (mean or 0)
        df = df.fillna(df.mean()).fillna(0)

        n_after = df.shape[1]

        if n_after < n_before:
            print(f"[sanitize_dataframe] Dropped {n_before - n_after} invalid columns from {name}")

        return df



    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        """
        Chronological split (67% train / 33% test).

        Returns X_train, X_test, y_train, y_test where:
          - X_* are DataFrames WITHOUT the target column 'log_return'
          - y_* are pd.Series named 'log_return'

        This is defensive against variants where self.features["log_return"] might be a 1-col DataFrame.
        """

        if "log_return" not in self.features.columns:
            raise ValueError("features DataFrame must contain 'log_return' column")

        n = len(self.features)
        split_idx = int(np.floor(n * (1 - 0.33)))  # keep same 67%-33% split as in both SARIMAX and VARMAX

        # Work on a copy & sanitize
        df = self.features.copy()
        df = self.sanitize_dataframe(df=df, name="features")

        # 1) Extract target as a Series (force Series even if df['log_return'] returns a DataFrame)
        #    Preserve original index and coerce to float.
        y_values = df["log_return"]

        # if it's a DataFrame with single column, extract that column
        if isinstance(y_values, pd.DataFrame) and y_values.shape[1] == 1:
            y_series = y_values.iloc[:, 0].astype(float).copy()

        else:
            # coerce to Series (this also handles numpy-array-likes)
            y_series = pd.Series(np.asarray(y_values).ravel(), index=df.index, name="log_return").astype(float)

        y_series.name = "log_return"  # enforce name

        # 2) Build X via the shared feature policy (drops log_return unless the
        #    experiment opts in), so this splitter cannot disagree with fit()/main.py.
        X = apply_target_policy(df, self.cfg).astype(float).copy()

        # 3) Split chronologically and return copies
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y_series.iloc[:split_idx].copy()
        y_test = y_series.iloc[split_idx:].copy()

        return X_train, X_test, y_train, y_test



    # ─── FITTING AND PREDICTING ───────────────────────────────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seq_len: Optional[int] = None,
            params: Optional[dict] = None, verbose: bool = True):

        """
        Train LSTM on provided training slices.
          - Defensive handling of X_train / y_train types and names.
          - Removes target column from X_train if present (prevents leakage).
          - Builds scalers from X_train / y_train, constructs sliding windows of length `seq_len`,
            and trains a small LSTM in PyTorch.

        Stores trained model and scalers on self.

        :param X_train: The pandas DataFrame (or array-like) containing the input features. May include 'log_return';
                        if present it will be dropped internally.
        :param y_train: The target (series/1-col DataFrame/array). Will be coerced to pd.Series named 'log_return'.
        :param seq_len: Sequence length (timesteps) for training windows.
        :param params: Hyperparameter overrides (dict).
        :param verbose: Print epoch losses if True.
        :return: The fitted LSTM model.
        """

        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided to fit()")

        # -------------------------
        # 1) Normalize y_train -> pd.Series named 'log_return' (preserve index if available)
        # -------------------------
        # If a single-column DataFrame was passed, extract the column
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]

        # If it's not a Series yet, try to make one and preserve index from X_train if possible
        if not isinstance(y_train, pd.Series):
            idx = None
            try:
                if hasattr(X_train, "index"):
                    idx = X_train.index

            except Exception:
                idx = None
            y_train = pd.Series(np.asarray(y_train).ravel(), index=idx, name="log_return")

        # Ensure dtype and name
        y_train = y_train.astype(float)
        y_train.name = "log_return"

        # -------------------------
        # 2) Normalize X_train -> pd.DataFrame without target column (if present)
        # -------------------------
        # If X_train is a numpy array, convert to DataFrame with integer column names (best-effort)
        if not isinstance(X_train, pd.DataFrame):
            try:
                X_train = pd.DataFrame(np.asarray(X_train))

            except Exception:
                raise ValueError("Cannot coerce X_train into DataFrame. Provide DataFrame or array-like.")

        # Feature policy: drop the target unless the experiment opts into feeding
        # past log_return as a feature (run #1). Single source of truth shared with
        # the orchestrator (main.py) via models.feature_policy.
        X_train = apply_target_policy(X_train, self.cfg)

        # If after the policy there's no feature column left, that's an error
        if X_train.shape[1] == 0:
            raise ValueError("X_train contains no features after dropping 'log_return'. Provide exogenous columns.")

        # Keep index alignment if y_train has an index
        if y_train is not None and hasattr(y_train, "index"):
            # try to align X_train and y_train index lengths if needed
            if len(X_train) != len(y_train):
                # prefer to reindex X_train to y_train index only if lengths match after reindex
                try:
                    X_train = X_train.reindex(y_train.index)

                except Exception:
                    # fallback: trim/pad to the min length
                    minlen = min(len(X_train), len(y_train))
                    X_train = X_train.iloc[:minlen]
                    y_train = y_train.iloc[:minlen]

        # -------------------------
        # 3) Base config and hyperparameter merging
        # -------------------------
        cfg = self.get_cfg()
        if params:
            cfg = {**cfg, **params}

        seq_len = int(seq_len or cfg["seq_len"])
        self.last_seq_len = seq_len

        hidden_size = int(cfg["hidden_size"])
        num_layers = int(cfg["num_layers"])
        dropout = float(cfg["dropout"])
        lr = float(cfg["lr"])
        batch_size = int(cfg["batch_size"])
        epochs = int(cfg["epochs"])

        # -------------------------
        # 4) Prepare numpy arrays and scalers
        # -------------------------
        # Sanitize BEFORE scaling. X_train can carry all-NaN macro columns
        # (e.g. us_consumer_price_index/us_core_cpi in the transformed
        # condition) and stray NaN. StandardScaler ignores NaN when fitting
        # but passes it straight through on transform, so the sliding windows
        # become NaN and training loss is NaN from epoch 1. ltc/cfc already
        # sanitize in fit(); lstm did not. Mirror them so windows are finite.
        X_train = self.sanitize_dataframe(X_train, name="X_train")

        # Record exact feature columns used for training (exogenous features only)
        self.feature_cols = list(X_train.columns)

        # Optionally precompute fill values (column means) to use for missing columns later:
        self.feature_fill = X_train.mean().to_dict()
        print(f"[LSTM.fit] Saved {len(self.feature_cols)} feature_cols")

        X_vals = X_train.values.astype(float)
        y_vals = y_train.values.reshape(-1, 1).astype(float)

        scaler_kind = cfg.get("scaler_kind", "zscore")
        self.scaler_X = _make_scaler(scaler_kind).fit(X_vals)
        self.scaler_y = _make_scaler(scaler_kind).fit(y_vals)

        X_scaled = self.scaler_X.transform(X_vals)
        y_scaled = self.scaler_y.transform(y_vals).ravel()

        # Build sliding windows (num_samples = n_rows - seq_len)
        n_samples = X_scaled.shape[0] - seq_len

        if n_samples <= 0:
            raise ValueError(f"Training data too short for seq_len={seq_len} (n_rows={X_scaled.shape[0]})")

        X_windows = np.stack([X_scaled[i: i + seq_len] for i in range(n_samples)])  # (n_samples, seq_len, n_features)
        y_targets = np.array([y_scaled[i + seq_len] for i in range(n_samples)], dtype=np.float32)

        # Ensure float32 numpy arrays (avoid unexpected object dtype)
        X_windows = np.asarray(X_windows, dtype=np.float32)
        y_targets = np.asarray(y_targets, dtype=np.float32)

        try:
            # preferred: zero-copy view of numpy memory
            X_windows_t = torch.from_numpy(X_windows)  # shape (n_samples, seq_len, n_features)
            y_targets_t = torch.from_numpy(y_targets)  # shape (n_samples,)

        except RuntimeError as e:
            # Defensive fallback: some PyTorch builds raise "Numpy is not available".
            msg = str(e)

            if "Numpy is not available" in msg or "numpy" in msg.lower():
                print("[LSTM] Warning: torch.from_numpy() failed, falling back to torch.tensor(list).")
                # convert to nested lists -> torch.tensor(...) (works without PyTorch numpy C-API)
                # Note: this uses extra memory and is slower, but is robust
                X_windows_t = torch.tensor(X_windows.tolist(), dtype=torch.float32)
                y_targets_t = torch.tensor(y_targets.tolist(), dtype=torch.float32)

            else:
                # re-raise unexpected RuntimeErrors
                raise

        ds = TensorDataset(X_windows_t, y_targets_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

        # create model
        n_features = X_scaled.shape[1]
        model = self.TorchLSTM(n_features=n_features, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # training loop
        start = time.time()
        model.train()

        for ep in range(1, epochs + 1):
            losses = []

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)  # (batch,)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if verbose:
                print(f"[LSTM] epoch {ep}/{epochs} loss={np.mean(losses):.6f}")

        self.runtime_seconds = time.time() - start
        self.model = model

        # At the end, store last_params from the merged cfg (for logging)
        self.last_params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "seq_len": seq_len
        }

        return model



    def predict_window(self, scaled_window: np.ndarray) -> float:

        """
        Predict single step from a scaled window array (seq_len, n_features).
        Returns de-scaled float.

        :param scaled_window:
        """

        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()

        with torch.no_grad():
            # Build tensor directly using torch.tensor (this avoids torch.from_numpy that may fail)
            x = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(
                self.device)  # (1, seq_len, n_features)
            y_tensor = self.model(x).cpu().detach()  # tensor on cpu

            # Extract python float without calling .numpy()
            if y_tensor.numel() == 1:
                y_scaled = float(y_tensor.item())

            else:
                # fallback for vector outputs
                y_scaled = float(y_tensor.squeeze().tolist()[0])

            # inverse transform using sklearn scaler (this uses numpy from Python side which is fine)
            y = float(self.scaler_y.inverse_transform(np.array([[y_scaled]]))[0, 0])

            return y



    def predict(self, model_fit: Optional[nn.Module] = None, X_train: Optional[pd.DataFrame] = None,
                X_test: Optional[pd.DataFrame] = None, y_train: Optional[pd.Series] = None,
                y_test: Optional[pd.Series] = None, include_intervals: bool = False):

        """
        Produce aligned 1-step-ahead predictions for the test horizon using true historical inputs.

        :param model_fit: optional model to use temporarily (will not overwrite self.model permanently)
        :param X_train: training features (chronological)
        :param X_test: test features (chronological)
        :param y_train: training target series
        :param y_test: test target series
        :param include_intervals: ignored (kept for API parity)
        :return: (y_pred_series, residuals_series, exp_id)
        """

        if X_test is None or y_test is None or X_train is None or y_train is None:
            raise ValueError("X_train, X_test, y_train, y_test must all be provided for aligned predictions")

        # ensure model and scalers exist
        if model_fit is None and self.model is None:
            raise ValueError("No trained model available. Call fit() first or provide model_fit.")

        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers not found. Ensure fit() was called on training data before predict().")

        # temporarily swap model if provided
        orig_model = self.model

        if model_fit is not None:
            self.model = model_fit

        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]

        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]

        if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
            # if they are numpy arrays, convert maintaining index where possible
            y_train = pd.Series(np.asarray(y_train).ravel(), index=(getattr(y_train, 'index', None) or None),
                                name="log_return")
            y_test = pd.Series(np.asarray(y_test).ravel(), index=(getattr(y_test, 'index', None) or None),
                               name="log_return")

        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        try:
            seq_len = int(self.last_seq_len or self.get_cfg()["seq_len"])

            # Build full chronological features matrix by concatenating train + test
            X_full = pd.concat([X_train, X_test]).astype(float).dropna(axis=0, how="all")
            # ensure same column order as X_test
            X_full = X_full[X_test.columns]

            # Ensure feature_cols exists
            if not hasattr(self, "feature_cols"):
                raise ValueError("feature_cols not recorded. Call fit() first.")

            # If you prefer to fill missing with column means you stored in fit():
            X_full = X_full.reindex(columns=self.feature_cols).fillna(pd.Series(self.feature_fill))

            X_full_vals = X_full.values
            X_full_scaled = self.scaler_X.transform(X_full_vals)

            train_size = len(X_train)
            preds = []
            test_positions = list(range(train_size, train_size + len(X_test)))

            for pos in test_positions:
                if pos - seq_len < 0:
                    preds.append(np.nan)

                else:
                    window = X_full_scaled[pos - seq_len: pos, :]
                    preds.append(self.predict_window(window))

            y_pred = pd.Series(preds, index=y_test.index, name="y_pred")
            residuals = y_test.astype(float) - y_pred

            # Per-fold metrics + logging are owned centrally by
            # main._log_folds_centrally / _fold_metrics. predict() keeps the
            # (y_pred, residuals, exp_id) tuple shape for the legacy
            # single-shot callers' signature.
            exp_id = None

            self.last_experiment_id = exp_id

            self.last_fold_id = None

            return y_pred, residuals, exp_id

        finally:
            # restore original model if we temporarily swapped it
            if model_fit is not None:
                self.model = orig_model



    # ─── HISTORIC AND FUTURE FORECASTING ──────────────────────────────────────────────────────────────────────────────

    # NOT BEING USED
    def historic_forecast(self, experiment_id: Optional[str] = None, include_intervals: bool = True,
                          X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None,
                          X_train: Optional[pd.DataFrame] = None, y_train: Optional[pd.Series] = None):

        """
        Aggregate in-sample metrics and log a single historic_forecast row (similar to SARIMAX wrapper).
        Requires same args as predict, forwards to predict().

        :param experiment_id: Uuid from experiment_results (if None, forecast() will try to use self.last_experiment_id;
                              if that is missing, it will raise).
        :param include_intervals: Whether to include prediction intervals (y_lower / y_upper).
        :param X_test: The pandas DataFrame containing the exog test data (X).
        :param y_test: The pandas DataFrame containing the endog test data (y).
        :param X_train: The pandas DataFrame containing the exog training data (X).
        :param y_train: The pandas DataFrame containing the endog training data (y).
        """

        if any(v is None for v in (X_test, y_test, X_train, y_train)):
            raise ValueError("X_train, X_test, y_train, y_test required for historic_forecast")

        y_pred, residuals, exp_id = self.predict(model_fit=self.model, X_train=X_train, X_test=X_test, y_train=y_train,
                                                 y_test=y_test, include_intervals=include_intervals)

        # aggregated metrics (reuse computations) - compute quick summary
        preds = y_pred.to_numpy(dtype=float)
        actuals = y_test.astype(float).to_numpy(dtype=float)
        finite_mask = np.isfinite(preds) & np.isfinite(actuals)

        if finite_mask.sum() == 0:
            mse = mae = rmse = smape = directional_accuracy = None

        else:
            p = preds[finite_mask]
            a = actuals[finite_mask]
            eps = 1e-9
            mse = float(np.mean((a - p) ** 2))
            mae = float(np.mean(np.abs(a - p)))
            rmse = float(np.sqrt(mse))
            smape = float(100.0 * np.mean(2.0 * np.abs(a - p) / (np.abs(a) + np.abs(p) + eps)))
            directional_accuracy = float(np.mean(np.sign(p) == np.sign(a)))

        historic_forecast_run_id = str(uuid.uuid4())

        row = {
            "experiment_id": exp_id,
            "forecast_run_id": historic_forecast_run_id,
            "ticker": self.ticker,
            "model_type": "LSTM",
            "horizon": int(len(y_test)),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "mase": None,
            "directional_accuracy": directional_accuracy,
            "coverage": None,
            "mean_pi_width": None,
            "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id", None) is not None else None,
            "notes": "LSTM run"
        }

        self.last_experiment_id = exp_id

        self.logger.log_historic_forecast(row)



    def future_forecast(self, experiment_id: Optional[str] = None, steps: Optional[int] = None,
                        include_intervals: bool = True, params: Optional[dict] = None):

        """
        Refit on the full available features and recursively forecast `steps` ahead.

        :param experiment_id: Uuid from experiment_results (if None, forecast() will try to use self.last_experiment_id;
                              if that is missing, it will raise).
        :param steps: Integer horizon. If None, tries to read from cfg at cfg["models"]["LSTM"]["forecast_steps"].
        :param include_intervals: Whether to include prediction intervals (y_lower / y_upper).
        :param params: The hyperparameters being used when doing a hyperparameter grid search.
        """

        cfg = self.get_cfg()

        # If no params passed, prefer to re-use the last trained params (useful during grid search)
        if params is None:
            params = self.last_params

        if steps is None:
            steps = int(cfg["forecast_steps"])

        full = self.features.copy().astype(float)

        if "log_return" not in full.columns:
            raise ValueError("features must include 'log_return'")

        X_cols = full.columns.tolist()
        X_full_df = full[X_cols].copy()
        y_full_series = pd.Series(full["log_return"].values, index=full.index, name="log_return")

        # refit on full series to get final model
        seq_len = int(self.last_seq_len or cfg["seq_len"])

        # pass params through to fit() if provided
        self.fit(X_full_df, y_full_series, seq_len=seq_len, params=params)

        # ensure scalers/model exist
        if self.scaler_X is None or self.scaler_y is None or self.model is None:
            raise ValueError("After refit scalers/model not available. Ensure fit() succeeded.")

        # build sliding window from last seq_len rows
        seq_len = int(self.last_seq_len or cfg["seq_len"])

        # Reindex full features to the training feature columns
        if not hasattr(self, "feature_cols"):
            raise ValueError("feature_cols not found. Ensure fit() succeeded before future_forecast().")

        # With means
        X_full_df = X_full_df.reindex(columns=self.feature_cols).fillna(pd.Series(self.feature_fill))

        # Recompute numpy values from the reindexed DataFrame (guaranteed to match scaler)
        X_full_df = X_full_df.reindex(columns=self.feature_cols)  # ensure order again (no-op if already right)
        X_full_vals = X_full_df.values.astype(float)

        # transform using the scaler fitted on the *same* columns/order
        X_full_scaled = self.scaler_X.transform(X_full_vals)

        if X_full_scaled.shape[0] < seq_len:
            raise ValueError("Not enough history to seed recursive forecast with seq_len=%d" % seq_len)

        last_window = X_full_scaled[-seq_len:].copy()  # (seq_len, n_features)

        # use one forecast_run_id for the whole batch
        future_forecast_run_id = str(uuid.uuid4())
        rows = []

        # create timestamp index for the forecast
        try:
            last_date = pd.Timestamp(full.index[-1])
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

        except Exception:
            future_idx = [pd.Timestamp(i).isoformat() for i in range(steps)]

        for i, ts in enumerate(future_idx):
            # predict next step from the current scaled window
            yhat = self.predict_window(last_window)

            # Build next_row from the last observed feature row of X_full_df
            # (X_full_df is already reindexed to self.feature_cols, so shapes match),
            # then replace no columns — the exogenous features do not contain 'log_return' by design.
            last_obs = X_full_df.iloc[-1].copy()  # pandas Series with index=self.feature_cols
            next_row = last_obs.values.astype(float)  # numpy array shape (n_features,)

            # If you *do* want to record the forecast back into a synthetic 'log_return' column
            # for bookkeeping, do that in a separate container (don't try to insert into next_row
            # because the model's features don't include log_return).
            # next_row_for_logging = next_row.copy()
            # (optionally) attach yhat to a separate dict if you need it.

            # scale next_row and append to sliding window
            next_row_scaled = self.scaler_X.transform(next_row.reshape(1, -1))[0]  # shape (n_features,)
            last_window = np.vstack([last_window[1:], next_row_scaled])

            # timestamp handling
            try:
                ts_str = pd.Timestamp(ts).to_pydatetime().isoformat()

            except Exception:
                ts_str = str(ts)

            row = {
                "experiment_id": experiment_id or getattr(self, "last_experiment_id", None),
                "forecast_run_id": future_forecast_run_id,
                "ticker": self.ticker,
                "model_type": "LSTM",
                "timestamp": ts_str,
                "horizon_step": int(i + 1),
                "y_forecast": float(yhat),
                "y_lower": None,
                "y_upper": None,
                "actual": None,
                "error": None,
                "abs_error": None,
                "sq_error": None,
                "direction_correct": None,
                "in_interval": None,
                "pi_width": None,
                "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id",
                                                                               None) is not None else None,
                "horizon": int(steps),
                "notes": "LSTM run"
            }

            rows.append(row)

        # log batch once
        self.last_experiment_id = experiment_id or getattr(self, "last_experiment_id", None)

        self.last_future_forecast_run_id = future_forecast_run_id

        self.logger.log_future_forecast(rows)