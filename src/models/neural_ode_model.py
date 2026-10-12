import time
import uuid
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from models.feature_policy import apply_target_policy


def _make_scaler(kind: str):
    if kind == "zscore":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler kind: {kind!r}. Use 'zscore' or 'minmax'.")


# ─── HELPER MODULES ───────────────────────────────────────────────────────────────────────────────────────────────────

class _ODEFunc(nn.Module):
    """
    dh/dt = MLP(h). Time-invariant ODE function — matches the Chen et al. (2018)
    'augmented neural ODE' shape with no explicit time dependence on the RHS.
    """

    def __init__(self, latent_size: int, ode_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, ode_hidden),
            nn.Tanh(),
            nn.Linear(ode_hidden, latent_size),
        )
        # Tanh-only init: Xavier on the linear weights.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, h):
        # t is a scalar tensor (required by torchdiffeq); ignored here.
        return self.net(h)


class _NeuralODENetwork(nn.Module):
    """
    GRU encoder over the window → h_0; ODE integration from t=0 to t=1; linear
    decoder reads h_1. Per the eng + CEO review decisions (GRU encoder; rk4
    fixed-step default with dopri5 grid combo). The lineage-faithful Chen et
    al. (2018) latent-ODE shape.
    """

    def __init__(self, n_features: int, latent_size: int, encoder_hidden: int,
                 ode_hidden: int, solver: str = "rk4", step_size: float = 0.25,
                 rtol: float = 1e-3, atol: float = 1e-4):
        super().__init__()
        # GRU encoder; encoder_hidden is the GRU hidden size, which we use as
        # latent_size directly. (Two knobs would over-parameterize the small
        #-data thesis setup.)
        self.encoder = nn.GRU(input_size=n_features, hidden_size=encoder_hidden,
                              num_layers=1, batch_first=True)
        # Project encoder output to latent_size (allows decoupling if desired).
        self.enc_proj = nn.Linear(encoder_hidden, latent_size)
        self.ode_func = _ODEFunc(latent_size=latent_size, ode_hidden=ode_hidden)
        self.decoder = nn.Linear(latent_size, 1)

        self.solver = solver
        self.step_size = float(step_size)
        self.rtol = float(rtol)
        self.atol = float(atol)

        nn.init.xavier_uniform_(self.enc_proj.weight)
        nn.init.zeros_(self.enc_proj.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, h_enc = self.encoder(x)          # h_enc: (1, batch, encoder_hidden)
        h0 = self.enc_proj(h_enc[0])        # (batch, latent_size)

        t = torch.linspace(0.0, 1.0, 2, device=x.device, dtype=x.dtype)
        if self.solver in ("rk4", "euler", "midpoint"):
            # Fixed-step solvers take step_size via options.
            h_traj = odeint(self.ode_func, h0, t,
                            method=self.solver,
                            options={"step_size": self.step_size})
        else:
            # Adaptive solvers (dopri5, dopri8, ...) use rtol/atol.
            h_traj = odeint(self.ode_func, h0, t,
                            method=self.solver, rtol=self.rtol, atol=self.atol)

        h1 = h_traj[-1]                     # (batch, latent_size)
        return self.decoder(h1).squeeze(-1)  # (batch,)


# ─── WRAPPER ──────────────────────────────────────────────────────────────────────────────────────────────────────────

class NeuralODEModel:
    """
    Neural ODE wrapper. Mirrors LSTMModel / CTRNNModel surface so it plugs into
    the walk-forward harness in src/main.py without special-casing.
    """

    def __init__(self, cfg: Dict, features: pd.DataFrame, logger, ticker: str):
        self.cfg = cfg
        self.features = features.copy()
        self.logger = logger
        self.ticker = ticker

        self.model: Optional[nn.Module] = None
        self.scaler_X = None
        self.scaler_y = None

        self.runtime_seconds: Optional[float] = None
        self.last_seq_len: Optional[int] = None
        self.last_params: Optional[dict] = None
        self.feature_cols: Optional[list] = None
        self.feature_fill: Optional[dict] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── HELPERS ──────────────────────────────────────────────────────────────────────────────────────────────────────

    def sanitize_dataframe(self, df: pd.DataFrame, name: str = "") -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        df = df.fillna(df.mean()).fillna(0)
        return df

    def get_cfg(self) -> dict:
        base = self.cfg.get("models", {}).get("NeuralODE", {})
        return {
            "latent_size": int(base.get("latent_size", 16)),
            "encoder_hidden": int(base.get("encoder_hidden", 32)),
            "ode_hidden": int(base.get("ode_hidden", 32)),
            "solver": str(base.get("solver", "rk4")),
            "step_size": float(base.get("step_size", 0.25)),
            "rtol": float(base.get("rtol", 1e-3)),
            "atol": float(base.get("atol", 1e-4)),
            "lr": float(base.get("lr", 5e-3)),
            "batch_size": int(base.get("batch_size", 32)),
            "epochs": int(base.get("epochs", 50)),
            "seq_len": int(base.get("seq_len", 20)),
            "grad_clip": float(base.get("grad_clip", 1.0)),
            "forecast_steps": int(base.get("forecast_steps", 10)),
            "scaler_kind": base.get("scaler_kind", "zscore"),
        }

    # ─── FIT ──────────────────────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, seq_len: Optional[int] = None,
            params: Optional[dict] = None, verbose: bool = False):
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided to fit()")

        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        if not isinstance(y_train, pd.Series):
            idx = getattr(X_train, "index", None)
            y_train = pd.Series(np.asarray(y_train).ravel(), index=idx, name="log_return")
        y_train = y_train.astype(float)
        y_train.name = "log_return"

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(np.asarray(X_train))
        # Feature policy: drop the target unless the experiment opts in (run #1).
        # Shared with main.py via models.feature_policy so both layers agree.
        X_train = apply_target_policy(X_train, self.cfg)
        if X_train.shape[1] == 0:
            raise ValueError("X_train contains no features after dropping 'log_return'.")

        if len(X_train) != len(y_train):
            try:
                X_train = X_train.reindex(y_train.index)
            except Exception:
                minlen = min(len(X_train), len(y_train))
                X_train = X_train.iloc[:minlen]
                y_train = y_train.iloc[:minlen]

        # Drop inf, drop all-NaN cols, fill remaining NaN with column means
        # so the scaler doesn't choke. See note in CTRNNModel.fit().
        X_train = self.sanitize_dataframe(X_train, name="X_train")
        if X_train.shape[1] == 0:
            raise ValueError("No usable feature columns after sanitization.")

        cfg = self.get_cfg()
        if params:
            cfg = {**cfg, **params}

        seq_len = int(seq_len or cfg["seq_len"])
        self.last_seq_len = seq_len

        latent_size = int(cfg["latent_size"])
        encoder_hidden = int(cfg["encoder_hidden"])
        ode_hidden = int(cfg["ode_hidden"])
        solver = str(cfg["solver"])
        step_size = float(cfg["step_size"])
        rtol = float(cfg["rtol"])
        atol = float(cfg["atol"])
        lr = float(cfg["lr"])
        batch_size = int(cfg["batch_size"])
        epochs = int(cfg["epochs"])
        grad_clip = float(cfg["grad_clip"])

        self.feature_cols = list(X_train.columns)
        self.feature_fill = X_train.mean().to_dict()

        X_vals = X_train.values.astype(float)
        y_vals = y_train.values.reshape(-1, 1).astype(float)

        scaler_kind = cfg.get("scaler_kind", "zscore")
        self.scaler_X = _make_scaler(scaler_kind).fit(X_vals)
        self.scaler_y = _make_scaler(scaler_kind).fit(y_vals)

        X_scaled = self.scaler_X.transform(X_vals)
        y_scaled = self.scaler_y.transform(y_vals).ravel()

        n_samples = X_scaled.shape[0] - seq_len
        if n_samples <= 0:
            raise ValueError(f"Training data too short for seq_len={seq_len} (n_rows={X_scaled.shape[0]})")

        X_windows = np.stack([X_scaled[i: i + seq_len] for i in range(n_samples)]).astype(np.float32)
        y_targets = np.array([y_scaled[i + seq_len] for i in range(n_samples)], dtype=np.float32)

        X_windows_t = torch.from_numpy(X_windows)
        y_targets_t = torch.from_numpy(y_targets)

        ds = TensorDataset(X_windows_t, y_targets_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        n_features = X_scaled.shape[1]
        model = _NeuralODENetwork(
            n_features=n_features, latent_size=latent_size,
            encoder_hidden=encoder_hidden, ode_hidden=ode_hidden,
            solver=solver, step_size=step_size, rtol=rtol, atol=atol,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        start = time.time()
        model.train()
        for ep in range(1, epochs + 1):
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                losses.append(loss.item())
            if verbose:
                print(f"[NeuralODE] epoch {ep}/{epochs} loss={np.mean(losses):.6f}")

        self.runtime_seconds = time.time() - start
        self.model = model

        self.last_params = {
            "latent_size": latent_size, "encoder_hidden": encoder_hidden,
            "ode_hidden": ode_hidden, "solver": solver, "step_size": step_size,
            "lr": lr, "batch_size": batch_size, "epochs": epochs,
            "seq_len": seq_len, "grad_clip": grad_clip,
        }
        return model

    # ─── PREDICT ──────────────────────────────────────────────────────────────────────────────────────────────────────

    def predict_window(self, scaled_window: np.ndarray) -> float:
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(self.device)
            y_scaled = float(self.model(x).cpu().item())
            return float(self.scaler_y.inverse_transform(np.array([[y_scaled]]))[0, 0])

    def predict(self, model_fit: Optional[nn.Module] = None, X_train: Optional[pd.DataFrame] = None,
                X_test: Optional[pd.DataFrame] = None, y_train: Optional[pd.Series] = None,
                y_test: Optional[pd.Series] = None, include_intervals: bool = False):
        # include_intervals: ignored (kept for API parity with LSTM/LTC/CFC)
        if X_test is None or y_test is None or X_train is None or y_train is None:
            raise ValueError("X_train, X_test, y_train, y_test must all be provided")
        if model_fit is None and self.model is None:
            raise ValueError("No trained model available. Call fit() first or provide model_fit.")
        if self.scaler_X is None or self.scaler_y is None:
            raise ValueError("Scalers not found. Ensure fit() was called before predict().")

        orig_model = self.model
        if model_fit is not None:
            self.model = model_fit

        try:
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
                y_train = y_train.iloc[:, 0]
            if isinstance(y_test, pd.DataFrame) and y_test.shape[1] == 1:
                y_test = y_test.iloc[:, 0]
            if not isinstance(y_train, pd.Series):
                y_train = pd.Series(np.asarray(y_train).ravel(), name="log_return")
            if not isinstance(y_test, pd.Series):
                y_test = pd.Series(np.asarray(y_test).ravel(), name="log_return")
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)

            seq_len = int(self.last_seq_len or self.get_cfg()["seq_len"])

            X_full = pd.concat([X_train, X_test]).astype(float)
            if self.feature_cols is None:
                raise ValueError("feature_cols not recorded. Call fit() first.")
            # Replace inf with NaN before fillna; otherwise scaler.transform chokes.
            X_full = (X_full.replace([np.inf, -np.inf], np.nan)
                            .reindex(columns=self.feature_cols)
                            .fillna(pd.Series(self.feature_fill))
                            .fillna(0.0))
            X_full_scaled = self.scaler_X.transform(X_full.values)

            train_size = len(X_train)
            preds = []
            for pos in range(train_size, train_size + len(X_test)):
                if pos - seq_len < 0:
                    preds.append(np.nan)
                else:
                    preds.append(self.predict_window(X_full_scaled[pos - seq_len: pos, :]))

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
            if model_fit is not None:
                self.model = orig_model

    # ─── FUTURE FORECAST ──────────────────────────────────────────────────────────────────────────────────────────────

    def future_forecast(self, experiment_id: Optional[str] = None, steps: Optional[int] = None,
                        include_intervals: bool = True, params: Optional[dict] = None):
        """
        Refit on the full available features and recursively forecast `steps` ahead.
        Parity-only; the walk-forward path does not call it.
        """
        cfg = self.get_cfg()
        if params is None:
            params = self.last_params
        if steps is None:
            steps = int(cfg["forecast_steps"])

        full = self.features.copy().astype(float)
        if "log_return" not in full.columns:
            raise ValueError("features must include 'log_return'")

        X_full_df = full.copy()
        y_full_series = pd.Series(full["log_return"].values, index=full.index, name="log_return")

        seq_len = int(self.last_seq_len or cfg["seq_len"])
        self.fit(X_full_df, y_full_series, seq_len=seq_len, params=params)

        X_full_df = X_full_df.reindex(columns=self.feature_cols).fillna(pd.Series(self.feature_fill))
        X_full_scaled = self.scaler_X.transform(X_full_df.values.astype(float))
        if X_full_scaled.shape[0] < seq_len:
            raise ValueError(f"Not enough history to seed recursive forecast with seq_len={seq_len}")

        last_window = X_full_scaled[-seq_len:].copy()
        future_forecast_run_id = str(uuid.uuid4())
        rows = []

        try:
            last_date = pd.Timestamp(full.index[-1])
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="B")
        except Exception:
            future_idx = list(range(steps))

        for i, ts in enumerate(future_idx):
            yhat = self.predict_window(last_window)
            last_obs = X_full_df.iloc[-1].values.astype(float)
            next_row_scaled = self.scaler_X.transform(last_obs.reshape(1, -1))[0]
            last_window = np.vstack([last_window[1:], next_row_scaled])

            try:
                ts_str = pd.Timestamp(ts).to_pydatetime().isoformat()
            except Exception:
                ts_str = str(ts)

            rows.append({
                "experiment_id": experiment_id or getattr(self, "last_experiment_id", None),
                "forecast_run_id": future_forecast_run_id,
                "ticker": self.ticker,
                "model_type": "NeuralODE",
                "timestamp": ts_str,
                "horizon_step": int(i + 1),
                "y_forecast": float(yhat),
                "y_lower": None, "y_upper": None, "actual": None,
                "error": None, "abs_error": None, "sq_error": None,
                "direction_correct": None, "in_interval": None, "pi_width": None,
                "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id", None) is not None else None,
                "horizon": int(steps),
                "notes": "NeuralODE run",
            })

        self.last_future_forecast_run_id = future_forecast_run_id
        self.logger.log_future_forecast(rows)
