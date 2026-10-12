import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX, VARMAXResults
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import uuid


class VARMAXModel:

    """
    Thin wrapper around statsmodels' VARMA(X) model to fit and produce both in-sample and out-of-sample forecasts.
    """

    def __init__(self, cfg, features, logger, ticker):

        """
        Constructor / Initialising the VARMA(X) model object.

        :param cfg: Your config.yaml file containing all settings.
        :param features: The features dataset we are forecasting on.
        :param logger: SupabaseLogger instance for loggers this models results properly.
        :param ticker: The ticker of the stock currently being analysed.
        """

        self.cfg = cfg
        self.features = features
        self.logger = logger
        self.ticker = ticker

        self.model_fit: Optional[VARMAXResults] = None
        self.runtime_seconds: Optional[float] = None

        self.last_experiment_id: Optional[str] = None
        self.last_fold_id: Optional[int] = None

        self.last_historic_forecast_run_id: Optional[str] = None
        self.last_future_forecast_run_id: Optional[str] = None

        # last_order will be a (p, q) tuple when set
        self.last_order: Optional[Tuple[int, int]] = None

        # target column: prefer explicit target_col, fallback to market_col (compat), then to 'log_return'
        var_cfg = self.cfg.get("models", {}).get("VARMAX", {})
        self.target_col = var_cfg.get("target_col") or var_cfg.get("market_col") or "log_return"



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def train_test_split(self):

        """
        Creating training and testing datasets.

        :return: Array containing the two (training and testing) datasets as pandas Series.
        """

        var_cfg = self.cfg.get("models", {}).get("VARMAX", {})
        endog_cols = var_cfg.get("endog_cols", None)

        # sensible default list (same as your earlier code). Keep the order stable.
        if endog_cols is None:
            endog_cols = [
                "log_return",
                "dji",
                "gld",
                "us_consumer_price_index",
                "income_q_gross_profit_pctchg",
            ]

        # select only existing columns
        available = [c for c in endog_cols if c in self.features.columns]

        if len(available) == 0:
            raise ValueError(f"No VARMAX endog columns found in features. Expected one of: {endog_cols}")

        endog = self.features[available].copy().astype(float)

        y_train, y_test = train_test_split(endog, test_size=0.33, shuffle=False)

        return y_train, y_test



    def ensure_target_series(self, df_like, target_col):

        """
        Return Series of target_col from df_like (DataFrame/Series).

        :param df_like:
        :param target_col:
        """

        if df_like is None:
            return None

        if isinstance(df_like, pd.Series):
            return df_like.astype(float)

        if isinstance(df_like, pd.DataFrame):
            if target_col in df_like.columns:
                return df_like[target_col].astype(float)

            if df_like.shape[1] == 1:
                return df_like.iloc[:, 0].astype(float)

            # fallback: squeeze if possible
            try:
                s = pd.Series(df_like.squeeze()).astype(float)
                return s

            except Exception:
                raise ValueError("Cannot extract target series from provided object.")

        # else
        raise ValueError("Unsupported type for extracting target series.")



    # ─── FITTING & PREDICTING ─────────────────────────────────────────────────────────────────────────────────────────

    def fit(self, y_train: pd.DataFrame = None, o: Optional[Tuple[int, int]] = None) -> VARMAXResults:

        """
        Fitting the statsmodel VARMA(X) model to the training data (y_train).

        :param y_train: pandas DataFrame containing the training data.
        :param o: If wanted a custom order for the VARMA(X) model to work with.
        :return: The fitted model.
        """

        if y_train is None:
            raise ValueError("y_train must be provided to fit()")

        if o is not None:
            order = tuple(o)

        else:
            order = tuple(self.cfg.get("models", {}).get("VARMAX", {}).get("order", (1, 0)))

        self.last_order = order

        enforce_stat = self.cfg.get("models", {}).get("VARMAX", {}).get("enforce_stationarity", True)
        enforce_inv = self.cfg.get("models", {}).get("VARMAX", {}).get("enforce_invertibility", True)

        model = VARMAX(endog=y_train, order=order, enforce_stationarity=enforce_stat, enforce_invertibility=enforce_inv)

        start_time = time()
        model_fit = model.fit(disp=False)
        runtime_seconds = time() - start_time
        print("VARMAX fit time:", runtime_seconds)

        try:
            print(model_fit.summary())

        except Exception:
            pass

        self.runtime_seconds = runtime_seconds
        self.model_fit = model_fit

        return model_fit



    def predict(self, model_fit: VARMAXResults = None, y_test: pd.DataFrame = None,
                y_train: pd.DataFrame = None, include_intervals: bool = False) -> tuple:

        """
        Predicts the test data based on the fitted model.
        If y_test is provided and has an index, use get_prediction(start, end) to produce aligned preds and
        (optionally) conf_int.

        :param model_fit: The fitted model.
        :param y_test: The pandas DataFrame containing the test data (actuals).
        :param y_train: The pandas Series used for training (used for MASE scale).
        :param include_intervals: If True try to return conf_int as well (not required).
        :return: (predictions Series, residuals Series, optional conf_int DataFrame)
        """

        eps = 1e-9

        if model_fit is None:
            model_fit = self.model_fit

        if model_fit is None:
            raise ValueError("No fitted model provided to predict(). Call fit() first or provide model_fit.")

        conf_int = None

        if y_test is None:
            raise ValueError("y_test must be provided to predict()")

        # Rolling 1-step-ahead within the test fold, matching the SARIMAX path so
        # classical and neural models use the same protocol when re-enabled.
        # NOTE: this path is currently DEAD CODE. VARMAX is dropped from
        # experiment.runs in config.yaml because the walk-forward harness at
        # main.py:772 feeds VARMAX a univariate Series, but statsmodels.VARMAX
        # requires multivariate endog and crashes in fit() before predict() is
        # reached. The shape handling below is kept multivariate-ready so this
        # path will work once walk-forward builds the proper endog frame (see
        # endog_cols in this file's train_test_split, line 80-87).
        # No silent fallback to multi-step forecasts.
        y_test_target = self.ensure_target_series(y_test, self.target_col)
        if isinstance(y_test, pd.DataFrame):
            endog_cols = list(y_test.columns)
        else:
            endog_cols = [getattr(y_test, "name", None) or self.target_col]

        results = model_fit
        preds = []
        for pos in range(len(y_test_target)):
            f = results.forecast(steps=1)
            if isinstance(f, pd.DataFrame):
                pred_val = float(f.iloc[0][self.target_col]) if self.target_col in f.columns \
                    else float(f.iloc[0, 0])
            elif isinstance(f, pd.Series):
                pred_val = float(f.iloc[0])
            else:
                arr = np.asarray(f).reshape(1, -1)
                if self.target_col in endog_cols:
                    pred_val = float(arr[0, endog_cols.index(self.target_col)])
                else:
                    pred_val = float(arr[0, 0])
            preds.append(pred_val)

            # Build the realized row for append. For multivariate endog we need
            # the full row; for univariate we just need the scalar.
            if isinstance(y_test, pd.DataFrame):
                row = y_test.iloc[pos].values.astype(float)
            else:
                row = np.asarray([float(y_test_target.iloc[pos])])
            if not np.all(np.isfinite(row)):
                # Fail the fold loudly. Silently skipping append would freeze
                # the state and misalign every subsequent prediction.
                raise ValueError(
                    f"Non-finite row at test position {pos}: {row}. "
                    "Refusing to advance the rolling state with NaN/inf."
                )
            if row.shape[0] == 1:
                results = results.append([float(row[0])], refit=False)
            else:
                results = results.append(row.reshape(1, -1), refit=False)

        y_pred_df = pd.DataFrame({self.target_col: preds}, index=y_test.index)

        # extract target series for metrics
        y_pred_target = self.ensure_target_series(y_pred_df, self.target_col)
        y_test_target = self.ensure_target_series(y_test, self.target_col)

        # residuals (target only)
        residuals = y_test_target - y_pred_target
        residuals = pd.Series(residuals, index=y_test.index, name="residuals")

        # compute metrics on 1-D arrays
        y_pred_arr = np.asarray(y_pred_target).astype(float)
        y_test_arr = np.asarray(y_test_target).astype(float)

        if y_pred_arr.shape[0] != y_test_arr.shape[0]:
            minlen = min(y_pred_arr.shape[0], y_test_arr.shape[0])
            y_pred_arr = y_pred_arr[:minlen]
            y_test_arr = y_test_arr[:minlen]

        finite_mask = np.isfinite(y_pred_arr) & np.isfinite(y_test_arr)

        if finite_mask.sum() == 0:
            mse = mae = rmse = directional_accuracy = smape = None
            print("Warning: no finite prediction/actual pairs available for metric calculation.")

        else:
            yp = y_pred_arr[finite_mask]
            yt = y_test_arr[finite_mask]
            mse = float(np.mean((yt - yp) ** 2))
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(mse))
            directional_accuracy = float(np.mean(np.sign(yp) == np.sign(yt)))
            smape = float(100.0 * np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + eps)))

        # MASE scale using y_train target series
        if y_train is not None:
            train_target = self.ensure_target_series(y_train, self.target_col)

            if train_target is not None and len(train_target) >= 2:
                scale_errors = np.abs(train_target.iloc[1:].values - train_target.iloc[:-1].values)
                scale = float(np.mean(scale_errors)) if scale_errors.size > 0 else eps
                mase = float(mae / (scale + eps)) if mae is not None else None

            else:
                mase = None

        else:
            mase = None

        print('MSE:', mse, 'MAE:', mae, 'RMSE:', rmse, 'SMAPE:', smape, 'MASE:', mase)

        # Per-fold metrics + logging are owned centrally by
        # main._log_folds_centrally / _fold_metrics. predict() keeps the
        # (y_pred, residuals, exp_id) tuple shape for the legacy
        # single-shot callers' signature.
        exp_id = None

        self.last_experiment_id = exp_id
        self.last_fold_id = None

        return y_pred_target, residuals, exp_id



    # ─── HISTORIC & FUTURE FORECASTING ────────────────────────────────────────────────────────────────────────────────

    def historic_forecast(self, experiment_id: Optional[str] = None, include_intervals: bool = True,
                          y_test: Optional[pd.Series] = None):

        """
        Produces aggregated in-sample forecast for the entire y_test horizon and logs them as a single row to Supabase.

        :param experiment_id: Uuid from experiment_results (if None, forecast() will try to use self.last_experiment_id;
                              if that is missing, it will raise).
        :param include_intervals: Whether to include prediction intervals (y_lower / y_upper).
        :param y_test: The test data. This method will calculate error metrics based on it if available.
        """

        historic_forecast_run_id = str(uuid.uuid4())
        self.last_historic_forecast_run_id = historic_forecast_run_id

        if self.model_fit is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        if experiment_id is None:
            experiment_id = getattr(self, "last_experiment_id", None)

            if experiment_id is None:
                raise ValueError("experiment_id not provided and no last_experiment_id found.")

        if y_test is None:
            raise ValueError("y_test dataset must be provided!")

        steps = len(y_test)

        try:
            forecast_res = self.model_fit.get_forecast(steps=steps)
            y_forecast_df = forecast_res.predicted_mean

            if hasattr(y_forecast_df, "index"):
                y_forecast_df = y_forecast_df.reindex(y_test.index)

                if y_forecast_df.isna().any().any():
                    fallback = pd.DataFrame(self.model_fit.forecast(len(y_test)), index=y_test.index,
                                            columns=y_test.columns)
                    y_forecast_df = y_forecast_df.fillna(fallback)

            else:
                y_forecast_df = pd.DataFrame(y_forecast_df, index=y_test.index, columns=y_test.columns)

            conf_int = forecast_res.conf_int() if include_intervals else None

            if conf_int is not None and hasattr(conf_int, "index"):
                conf_int = conf_int.reindex(y_test.index)

        except Exception:
            y_forecast_df = pd.DataFrame(self.model_fit.forecast(steps), index=y_test.index, columns=y_test.columns)
            conf_int = None

        # metrics on target
        preds = self.ensure_target_series(y_forecast_df, self.target_col).to_numpy(dtype=float)
        actuals = self.ensure_target_series(y_test, self.target_col).to_numpy(dtype=float)

        lowers = np.full_like(preds, np.nan)
        uppers = np.full_like(preds, np.nan)

        if conf_int is not None:
            try:
                if isinstance(conf_int.columns, pd.MultiIndex):
                    lowers = conf_int[(self.target_col, "lower")].to_numpy(dtype=float)
                    uppers = conf_int[(self.target_col, "upper")].to_numpy(dtype=float)

                else:
                    lowers = conf_int.iloc[:, 0].to_numpy(dtype=float)
                    uppers = conf_int.iloc[:, 1].to_numpy(dtype=float)

            except Exception:
                lowers = np.full_like(preds, np.nan)
                uppers = np.full_like(preds, np.nan)

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

        interval_mask = np.isfinite(lowers) & np.isfinite(uppers) & finite_mask

        if interval_mask.sum() == 0:
            coverage = mean_pi_width = None

        else:
            L = lowers[interval_mask];
            U = uppers[interval_mask];
            A = actuals[interval_mask]
            coverage = float(np.mean((A >= L) & (A <= U)))
            mean_pi_width = float(np.mean(U - L))

        # MASE on the target column in the full series
        try:
            full_series = self.features[self.target_col].copy()
            first_test_ts = y_test.index[0]

            if first_test_ts in full_series.index:
                train_series = full_series.loc[:first_test_ts].iloc[:-1]

            else:
                train_series = full_series.iloc[:-len(y_test)]

            if len(train_series) >= 2:
                scale_errors = np.abs(train_series.iloc[1:].values - train_series.iloc[:-1].values)
                scale = float(np.mean(scale_errors)) if scale_errors.size > 0 else 1e-9
                mase = float(mae / (scale + 1e-9)) if mae is not None else None

            else:
                mase = None

        except Exception:
            mase = None

        row = {
            "experiment_id": experiment_id,
            "forecast_run_id": historic_forecast_run_id,
            "ticker": self.ticker,
            "model_type": "VARMAX",
            "horizon": int(steps),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "mase": mase,
            "directional_accuracy": directional_accuracy,
            "coverage": coverage,
            "mean_pi_width": mean_pi_width,
            "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id",
                                                                           None) is not None else None,
            "notes": "VARMAX run"
        }

        self.last_experiment_id = experiment_id

        inserted = self.logger.log_historic_forecast(row)



    def future_forecast(self, experiment_id: Optional[str] = None, steps: Optional[int] = None,
                        include_intervals: bool = True, o: Optional[Tuple[int, int]] = None):

        """
        Produces out-of-sample forecasts for a given horizon (steps) and logs them to Supabase.

        :param experiment_id: Uuid from experiment_results (if None, forecast() will try to use self.last_experiment_id;
                              if that is missing, it will raise).
        :param steps: Integer horizon. If None, tries to read from cfg at cfg["models"]["VARMAX"]["forecast_steps"].
        :param include_intervals: Whether to include prediction intervals (y_lower / y_upper).
        :param o: The order being used when doing a hyperparameter grid search.
        """

        future_forecast_run_id = str(uuid.uuid4())
        self.last_future_forecast_run_id = future_forecast_run_id

        # build the full endog DataFrame (same columns as train_test_split)
        var_cfg = self.cfg.get("models", {}).get("VARMAX", {})
        endog_cols = var_cfg.get("endog_cols", None)

        if endog_cols is None:
            endog_cols = [c for c in self.features.columns if
                          c in [self.target_col, "dji", "gld", "us_consumer_price_index",
                                "income_q_gross_profit_pctchg"]]

        full_endog = self.features[endog_cols].copy().dropna(axis=0, how="all").astype(float)

        # determine order: explicit arg -> last_order -> cfg default
        order_to_use = tuple(o) if o is not None else (
                    getattr(self, "last_order", None) or tuple(var_cfg.get("order", (1, 0))))

        # re-fit on full series
        future_model_fit = self.fit(full_endog, order_to_use)

        # determine steps
        if steps is None:
            try:
                steps = int(var_cfg.get("forecast_steps", 10))

            except Exception:
                steps = 10

        if experiment_id is None:
            experiment_id = getattr(self, "last_experiment_id", None)

            if experiment_id is None:
                raise ValueError("experiment_id not provided and no last_experiment_id found.")

        forecast_res = future_model_fit.get_forecast(steps=steps)
        y_forecast_df = forecast_res.predicted_mean

        conf_int = None

        if include_intervals:
            try:
                conf_int = forecast_res.conf_int()

            except Exception:
                conf_int = None

        # index for forecast timestamps
        if hasattr(y_forecast_df, "index") and not isinstance(y_forecast_df.index, pd.RangeIndex):
            idx = y_forecast_df.index

        else:
            last_date = pd.Timestamp(full_endog.index[-1])
            idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

        # ensure DataFrame-like
        if not isinstance(y_forecast_df, pd.DataFrame):
            y_forecast_df = pd.DataFrame(y_forecast_df, index=idx, columns=full_endog.columns)

        rows = []

        for i, ts in enumerate(idx):
            ts_str = pd.Timestamp(ts).to_pydatetime().isoformat() if isinstance(ts, (pd.Timestamp, datetime)) else str(
                ts)

            try:
                forecast_val = float(y_forecast_df.iloc[i][self.target_col])

            except Exception:
                forecast_val = float(y_forecast_df.iloc[i, 0])

            lower = None
            upper = None

            if conf_int is not None:
                try:
                    if isinstance(conf_int.columns, pd.MultiIndex):
                        lower = float(conf_int.iloc[i][(self.target_col, "lower")])
                        upper = float(conf_int.iloc[i][(self.target_col, "upper")])

                    else:
                        lower = float(conf_int.iloc[i, 0])
                        upper = float(conf_int.iloc[i, 1])

                except Exception:
                    lower = None
                    upper = None

            pi_width = (upper - lower) if (lower is not None and upper is not None) else None

            row = {
                "experiment_id": experiment_id,
                "forecast_run_id": future_forecast_run_id,
                "ticker": self.ticker,
                "model_type": "VARMAX",
                "timestamp": ts_str,
                "horizon_step": int(i + 1),
                "y_forecast": forecast_val,
                "y_lower": float(lower) if lower is not None else None,
                "y_upper": float(upper) if upper is not None else None,
                "actual": None,
                "error": None,
                "abs_error": None,
                "sq_error": None,
                "direction_correct": None,
                "in_interval": None,
                "pi_width": float(pi_width) if pi_width is not None else None,
                "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id",
                                                                               None) is not None else None,
                "horizon": int(steps),
                "notes": "VARMAX run"
            }

            rows.append(row)

        self.last_experiment_id = experiment_id

        inserted = self.logger.log_future_forecast(rows)



    # ─── VISUALISING RESULTS ──────────────────────────────────────────────────────────────────────────────────────────

    def results_vis(self, predictions: pd.Series = None, residuals: pd.Series = None):

        """
        Visualising the results via the matplotlib.pyplot library.

        :param predictions:
        :param residuals:
        """

        plt.figure(figsize=(10, 4))
        plt.plot(residuals)
        plt.axhline(0, linestyle='--', color='k')
        plt.title('Residuals from VARMAX Model', fontsize=16)
        plt.ylabel('Error', fontsize=12)
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/VARMAX/Residuals/residuals.png")

        plt.figure(figsize=(10, 4))
        plt.plot(self.features[self.target_col])
        plt.plot(predictions)
        plt.legend(('Data', 'Predictions'), fontsize=12)
        plt.title(f'{self.ticker} — {self.target_col} and predictions', fontsize=14)
        plt.ylabel(self.target_col, fontsize=12)
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/VARMAX/Predictions/predictions.png")