import time
import uuid
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.feature_policy import apply_target_policy


def _make_scaler(kind: str):
    if kind == "zscore":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler kind: {kind!r}. Use 'zscore' or 'minmax'.")
from ncps.wirings import AutoNCP
from ncps.torch import CfC



class SequenceLearner(pl.LightningModule):

    """
    LightningModule for training a RNNSequence module.
    """

    def __init__(self, model, lr=0.005):

        """

        :param model:
        :param lr:
        """

        super().__init__()

        self.model = model
        self.lr = lr



    def training_step(self, batch, batch_idx):

        """

        :param batch:
        :param batch_idx:
        :return:
        """

        x, y = batch
        y_hat, _ = self.model.forward(x)

        # Take only the last timestep prediction for sequence-to-one forecasting
        y_hat = y_hat[:, -1, :].squeeze(-1)  # shape [batch]

        # DEBUG: uncomment to inspect shapes once
        print(f"[DEBUG][train] x:{tuple(x.shape)} y:{tuple(y.shape)} y_hat:{tuple(y_hat.shape)}")

        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        return {"loss": loss}



    def validation_step(self, batch, batch_idx):

        """

        :param batch:
        :param batch_idx:
        :return:
        """

        x, y = batch
        y_hat, _ = self.model.forward(x)

        # Match training behavior: last-timestep -> seq2one
        y_hat = y_hat[:, -1, :].squeeze(-1)  # shape [batch]

        # DEBUG: uncomment to inspect shapes once
        print(f"[DEBUG][val] x:{tuple(x.shape)} y:{tuple(y.shape)} y_hat:{tuple(y_hat.shape)}")

        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        return loss



    def test_step(self, batch, batch_idx):

        """
        Here we just reuse the validation_step for testing

        :param batch:
        :param batch_idx:
        :return:
        """

        return self.validation_step(batch, batch_idx)



    def configure_optimizers(self):

        """

        :return:
        """

        return torch.optim.Adam(self.model.parameters(), lr=self.lr)



class CfCModel:

    """
    Thin wrapper around NCPs CfC model to fit and produce both in-sample and out-of-sample forecasts.
    """

    def __init__(self, cfg: Dict, features: pd.DataFrame, logger, ticker: str):

        """
        Constructor / Initialising NCPs CfC model.

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
        self.out_features = 1
        self.n_features = None

        self.runtime_seconds: Optional[float] = None
        self.last_seq_len: Optional[int] = None
        self.last_params: Optional[dict] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def get_cfg(self) -> dict:

        """
        Returns CfC hyperparameters with sensible defaults.
        """

        base = self.cfg.get("models", {}).get("CfC", {})

        return {
            "units": int(base.get("units", 16)),
            "lr": float(base.get("lr", 0.01)),
            "epochs": int(base.get("epochs", 10)),
            "seq_len": int(base.get("seq_len", 20)),
            "clip_gradient": int(base.get("clip_gradient", 1)),
            "batch_first": int(base.get("batch_first", True)),
            "batch_size": int(base.get("batch_size", 32)),
            "sparsity_level": float(base.get("sparsity_level", 0.5)),
            "forecast_steps": int(base.get("forecast_steps", 10))
        }



    def sanitize_dataframe(self, df, name: Optional[str] = ""):

        """
        Drop or repair invalid columns before model training.

        :param df: The pandas Dataframe being sanitized before fitting the model.
        :param name: (Optional) The name of the dataset being sanitized.
        """

        # Remember original type to return Series if appropriate
        was_series = isinstance(df, (pd.Series,))

        # Convert ndarray/Series -> DataFrame for column-wise ops
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        elif isinstance(df, pd.Series):
            df = df.to_frame()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("sanitize_dataframe expects DataFrame/Series/ndarray")

        # Defensive conversions & replace infs
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop columns that are completely empty after inf->nan
        df = df.dropna(axis=1, how="all")

        # Fill remaining NaNs with column mean, then 0 as fallback
        df = df.fillna(df.mean()).fillna(0)

        # Clip extreme magnitudes (tweak threshold as needed)
        max_abs = 1e100
        df = df.clip(lower=-max_abs, upper=max_abs)

        # If caller passed a Series originally and result has 1 column, return Series
        if was_series and df.shape[1] == 1:
            s = df.iloc[:, 0]
            s.name = getattr(df, "columns", [None])[0] or "log_return"
            return s

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
        df = self.sanitize_dataframe(df=df)

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
        Train CfC on provided training slices.
          - Defensive handling of X_train / y_train types and names.
          - Removes target column from X_train if present (prevents leakage).

        :param X_train: The pandas DataFrame (or array-like) containing the input features. May include 'log_return';
                        if present it will be dropped internally.
        :param y_train: The target (series/1-col DataFrame/array). Will be coerced to pd.Series named 'log_return'.
        :param seq_len: Sequence length (timesteps) for training windows.
        :param params: Hyperparameter overrides (dict).
        :param verbose: Print epoch losses if True.
        :return: The fitted CfC model.
        """

        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided to fit()")

        X_train = self.sanitize_dataframe(df=X_train)
        y_train = self.sanitize_dataframe(df=y_train)

        # 1) Normalize y_train -> pd.Series named 'log_return' (preserve index if available)
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

        # 2) Normalize X_train -> pd.DataFrame without target column (if present)
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

        # 3) Base config and hyperparameter merging
        cfg = self.get_cfg()
        if params:
            cfg = {**cfg, **params}

        seq_len = int(seq_len or cfg["seq_len"])
        self.last_seq_len = seq_len

        units = int(cfg["units"])
        lr = float(cfg["lr"])
        batch_first = bool(cfg["batch_first"])
        clip_gradient = float(cfg["clip_gradient"])
        sparsity_level = float(cfg["sparsity_level"])
        batch_size = int(cfg["batch_size"])
        epochs = int(cfg["epochs"])

        # 4) Prepare numpy arrays and scalers
        # Record exact feature columns used for training (exogenous features only)
        self.feature_cols = list(X_train.columns)

        # Optionally precompute fill values (column means) to use for missing columns later:
        self.feature_fill = X_train.mean().to_dict()
        print(f"[CfC.fit] Saved {len(self.feature_cols)} feature_cols")

        X_vals = X_train.values.astype(float)
        y_vals = y_train.values.reshape(-1, 1).astype(float)

        print("=== Checking X_vals before scaling ===")
        inf_cols_idx = np.where(np.isinf(X_vals).any(axis=0))[0]
        nan_cols_idx = np.where(np.isnan(X_vals).any(axis=0))[0]
        large_val_threshold = 1e100
        large_cols_idx = np.where((np.abs(X_vals) > large_val_threshold).any(axis=0))[0]

        print("Columns containing infinity (by index):", inf_cols_idx)
        print("Columns containing NaN (by index):", nan_cols_idx)
        print("Columns containing extremely large values (by index):", large_cols_idx)

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
                print("[CfC] Warning: torch.from_numpy() failed, falling back to torch.tensor(list).")
                # convert to nested lists -> torch.tensor(...) (works without PyTorch numpy C-API)
                # Note: this uses extra memory and is slower, but is robust
                X_windows_t = torch.tensor(X_windows.tolist(), dtype=torch.float32)
                y_targets_t = torch.tensor(y_targets.tolist(), dtype=torch.float32)

            else:
                # re-raise unexpected RuntimeErrors
                raise

        data_X = torch.Tensor(X_windows_t)
        data_y = torch.Tensor(y_targets_t)

        ds = TensorDataset(data_X, data_y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

        # 5) Create model
        n_features = X_scaled.shape[1]
        self.n_features = n_features

        # Variant wiring/mode (run #2). Defaults reproduce #0/#1 exactly:
        # wiring="ncp" (AutoNCP sparse wiring) + mode="default" (ncps CfC default).
        wiring_kind = str(cfg.get("wiring", "ncp"))
        mode = str(cfg.get("mode", "default"))
        # mixed_memory (run #2 T3): wrap the CfC cell in an LSTM-augmented mixed-memory
        # RNN (ncps). Default False reproduces #0/#1 exactly. Parse as a real boolean
        # (bool("false") is True) so a stringy "false"/"0" override can't silently enable it.
        mixed_memory = str(cfg.get("mixed_memory", False)).strip().lower() in ("true", "1", "yes")

        if wiring_kind == "fc_backbone":
            # Fully-connected CfC with an MLP backbone. `units` is a plain int here,
            # so CfC emits `units` channels per step UNLESS proj_size is set. The
            # seq2one head does y_hat[:, -1, :].squeeze(-1) against a scalar target
            # (SequenceLearner, ~line 58), so we MUST project to out_features (=1) —
            # otherwise the loss compares a [batch, units] tensor to [batch] and
            # either errors or silently broadcasts into garbage gradients.
            backbone_units = int(cfg.get("backbone_units", 16))
            backbone_layers = int(cfg.get("backbone_layers", 1))
            model = CfC(input_size=n_features, units=units, proj_size=self.out_features,
                        mode=mode, backbone_units=backbone_units, backbone_layers=backbone_layers,
                        mixed_memory=mixed_memory, batch_first=batch_first).to(self.device)
            wiring_meta = {"wiring": "fc_backbone", "mode": mode,
                           "backbone_units": backbone_units, "backbone_layers": backbone_layers,
                           "mixed_memory": mixed_memory}
        else:
            wiring = AutoNCP(units=units, output_size=self.out_features, sparsity_level=sparsity_level)
            model = CfC(input_size=n_features, units=wiring, mode=mode,
                        mixed_memory=mixed_memory, batch_first=batch_first).to(self.device)
            wiring_meta = {"wiring": "ncp", "mode": mode, "mixed_memory": mixed_memory}

        learn = SequenceLearner(model, lr=lr)

        trainer = pl.Trainer(max_epochs=epochs, gradient_clip_val=clip_gradient, accelerator="cpu", devices=1,
                             enable_progress_bar=True, log_every_n_steps=5, num_sanity_val_steps=0, deterministic=True,
                             enable_model_summary=False)

        start = time.time()

        trainer.fit(model=learn, train_dataloaders=loader)

        self.model = model.to(self.device)

        self.last_params = {"units": units, "lr": lr, "batch_size": batch_size, "epochs": epochs, "seq_len": seq_len,
                            "sparsity_level": sparsity_level, "clip_gradient": clip_gradient, **wiring_meta}

        self.runtime_seconds = time.time() - start

        return model



    def predict_window(self, scaled_window: np.ndarray) -> float:

        """
        Predict single step from a scaled window array (seq_len, n_features).

        :param scaled_window:
        :return: De-scaled float.
        """

        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()

        with torch.no_grad():
            x = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_features)
            x = x.to(self.device)
            y_hat, _ = self.model(x)  # expect (1, seq_len, out_features)

            # seq2one: take last timestep, squeeze out features dim
            last = y_hat[:, -1, :].squeeze(-1).cpu().detach()  # shape (1,)

            if last.numel() == 1:
                y_scaled = float(last.item())

            else:
                y_scaled = float(last.squeeze().tolist()[0])

            # inverse scale
            y = float(self.scaler_y.inverse_transform(np.array([[y_scaled]]))[0, 0])

            return y



    def predict(self, model_fit: Optional[nn.Module] = None, X_train: Optional[pd.DataFrame] = None,
                X_test: Optional[pd.DataFrame] = None, y_train: Optional[pd.Series] = None,
                y_test: Optional[pd.Series] = None, include_intervals: bool = False):

        """
        Produce aligned 1-step-ahead predictions for the test horizon using true historical inputs.

        :param model_fit:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param include_intervals:
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

        # coerce y_train/y_test to Series
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]

        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]

        if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
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

            # ensure same column order as X_test (and consistent with training)
            if hasattr(X_test, "columns"):
                X_full = X_full.reindex(columns=list(X_test.columns))

            # require feature_cols recorded in fit()
            if not hasattr(self, "feature_cols"):
                raise ValueError("feature_cols not recorded. Call fit() first.")

            # reindex to training feature columns and fill with stored means
            X_full = X_full.reindex(columns=self.feature_cols).fillna(pd.Series(self.feature_fill))

            X_full_vals = X_full.values.astype(float)
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



    def future_forecast(self, experiment_id: Optional[str] = None, steps: Optional[int] = None,
                        include_intervals: bool = True, params: Optional[dict] = None):

        """
        Refit on the full available features and recursively forecast `steps` ahead.

        :param experiment_id:
        :param steps:
        :param include_intervals:
        :param params:
        """

        cfg = self.get_cfg()

        if params is None:
            params = self.last_params

        if steps is None:
            steps = int(cfg.get("forecast_steps", 10))

        full = self.features.copy().astype(float)

        if "log_return" not in full.columns:
            raise ValueError("features must include 'log_return'")

        X_cols = full.columns.tolist()
        X_full_df = full[X_cols].copy()
        y_full_series = pd.Series(full["log_return"].values, index=full.index, name="log_return")

        # refit on full series to get final model
        seq_len = int(self.last_seq_len or cfg["seq_len"])
        self.fit(X_full_df, y_full_series, seq_len=seq_len, params=params)

        if self.scaler_X is None or self.scaler_y is None or self.model is None:
            raise ValueError("After refit scalers/model not available. Ensure fit() succeeded.")

        # prepare last_window
        if not hasattr(self, "feature_cols"):
            raise ValueError("feature_cols not found. Ensure fit() succeeded before future_forecast().")

        X_full_df = X_full_df.reindex(columns=self.feature_cols).fillna(pd.Series(self.feature_fill))

        # === Defensive clean-up: replace infs, fill NaNs with stored feature_fill, and clip extremes ===
        # Replace +/-inf with NaN
        X_full_df = X_full_df.replace([np.inf, -np.inf], np.nan)

        # Ensure self.feature_fill has finite numbers for every feature
        for col in self.feature_cols:
            fv = self.feature_fill.get(col, 0)

            if not np.isfinite(fv):
                # fallback to 0 if stored fill is non-finite
                self.feature_fill[col] = 0.0

        # Fill NaNs using the stored means (feature_fill), then final fallback 0
        X_full_df = X_full_df.fillna(pd.Series(self.feature_fill)).fillna(0)

        # Clip extreme values to avoid overflow in scaler / numpy
        clip_thresh = 1e100
        X_full_df = X_full_df.clip(lower=-clip_thresh, upper=clip_thresh)

        # Diagnostic printing: map problematic columns (if any) to names
        Xvals = X_full_df.values
        inf_idx = np.where(np.isinf(Xvals).any(axis=0))[0]
        nan_idx = np.where(np.isnan(Xvals).any(axis=0))[0]
        large_idx = np.where((np.abs(Xvals) > clip_thresh).any(axis=0))[0]

        if len(inf_idx) or len(nan_idx) or len(large_idx):
            print("future_forecast: problematic columns detected BEFORE transform:")
            print("  inf indices:", inf_idx, "names:", [self.feature_cols[i] for i in inf_idx])
            print("  nan indices:", nan_idx, "names:", [self.feature_cols[i] for i in nan_idx])
            print("  large indices:", large_idx, "names:", [self.feature_cols[i] for i in large_idx])

            # As a last resort, replace any remaining non-finite values by feature_fill or 0:
            X_full_df = X_full_df.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(self.feature_fill)).fillna(0)

        X_full_vals = X_full_df.values.astype(float)
        X_full_scaled = self.scaler_X.transform(X_full_vals)

        if X_full_scaled.shape[0] < seq_len:
            raise ValueError("Not enough history to seed recursive forecast with seq_len=%d" % seq_len)

        last_window = X_full_scaled[-seq_len:].copy()  # (seq_len, n_features)

        future_forecast_run_id = str(uuid.uuid4())
        rows = []

        # create timestamp index for the forecast
        try:
            last_date = pd.Timestamp(full.index[-1])
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

        except Exception:
            future_idx = [pd.Timestamp(i).isoformat() for i in range(steps)]

        for i, ts in enumerate(future_idx):
            yhat = self.predict_window(last_window)

            last_obs = X_full_df.iloc[-1].copy()
            next_row = last_obs.values.astype(float)

            next_row_scaled = self.scaler_X.transform(next_row.reshape(1, -1))[0]
            last_window = np.vstack([last_window[1:], next_row_scaled])

            try:
                ts_str = pd.Timestamp(ts).to_pydatetime().isoformat()

            except Exception:
                ts_str = str(ts)

            row = {
                "experiment_id": experiment_id or getattr(self, "last_experiment_id", None),
                "forecast_run_id": future_forecast_run_id,
                "ticker": self.ticker,
                "model_type": "CfC",
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
                "fold_id": int(getattr(self, "last_fold_id", None))
                           if getattr(self, "last_fold_id", None) is not None else None,
                "horizon": int(steps),
                "notes": "CfC Future Forecast Run"
            }

            rows.append(row)

        self.last_experiment_id = experiment_id or getattr(self, "last_experiment_id", None)

        self.last_future_forecast_run_id = future_forecast_run_id

        self.logger.log_future_forecast(rows)