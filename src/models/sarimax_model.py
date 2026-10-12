import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from typing import Optional, List
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import uuid



class SARIMAModel:

    """
    Thin wrapper around statsmodels' SARIMAX model to fit and produce both in-sample and out-of-sample forecasts.
    """

    def __init__(self, cfg, features, logger, ticker):

        """
        Constructor / Initialising the SARIMAX model object.
        
        :param cfg: Your config.yaml file containing all settings.
        :param features: The features dataset we are forecasting on.
        :param logger: SupabaseLogger instance for loggers this models results properly.
        :param ticker: The ticker of the stock currently being analysed.
        """

        self.cfg = cfg
        self.features = features
        self.logger = logger
        self.ticker = ticker

        self.model_fit: Optional[SARIMAXResults] = None
        self.runtime_seconds: Optional[float] = None

        self.last_experiment_id: Optional[str] = None
        self.last_fold_id: Optional[int] = None

        self.last_historic_forecast_run_id: Optional[str] = None
        self.last_future_forecast_run_id: Optional[str] = None

        self.last_order: Optional[List] = None
        self.last_seasonal_order: Optional[List] = None



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def train_test_split(self):

        """
        Creating training and testing datasets.

        :return: Array containing the two (training and testing) datasets as pandas Series.
        """

        endog = self.features["log_return"].copy()
        endog.name = "log_return"

        y_train, y_test = train_test_split(endog, test_size=0.33, shuffle=False)

        return y_train, y_test



    # ─── FITTING AND PREDICTING ───────────────────────────────────────────────────────────────────────────────────────

    def fit(self, y_train: pd.DataFrame = None, o: Optional[List] = None, so: Optional[List] = None) -> SARIMAXResults:

        """
        Fitting the statsmodel SARIMA(X) model to the training data (y_train).

        :param y_train: pandas DataFrame containing the training data.
        :return: The fitted model.
        """

        if o is not None:
            order = tuple(o)
        else:
            order = tuple(self.cfg["models"]["SARIMAX"]["order"])
        self.last_order = order

        if so is not None:
            seasonal_order = tuple(so)
        else:
            seasonal_order = tuple(self.cfg["models"]["SARIMAX"]["seasonal_order"])
        self.last_seasonal_order = seasonal_order

        stationarity = self.cfg["models"]["SARIMAX"].get("enforce_stationarity", True)
        invertibility = self.cfg["models"]["SARIMAX"].get("enforce_invertibility", True)

        model = SARIMAX(endog=y_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=stationarity, enforce_invertibility=invertibility)

        start_time = time()
        model_fit = model.fit(disp=False)
        runtime_seconds = time() - start_time
        print('Model Fitting Time:', runtime_seconds)

        print(model_fit.summary())

        self.runtime_seconds = runtime_seconds
        self.model_fit = model_fit

        return model_fit



    def fit_auto(self, y_train: pd.DataFrame = None, p_max: int = 2, q_max: int = 2,
                 d: int = 0) -> SARIMAXResults:

        """
        Per-fold AIC order search for the ``sarimax_aic`` baseline (run #2).

        Fits every (p, d, q) in {0..p_max} x {d} x {0..q_max} with a fixed
        ``seasonal_order=(0,0,0,0)`` and ``trend="n"`` (no constant, matching the
        fixed-order AR(1) baseline), on train-only data, and keeps the lowest-AIC
        fit.

        Each candidate is GUARDED: a fit that raises (LinAlgError / non-invertible)
        OR returns a non-finite AIC is skipped. If every candidate fails, raises
        ``RuntimeError`` so the walk-forward harness marks the fold failed (it then
        drops from the aligned DM set, same as any model failure). statsmodels
        convergence chatter is suppressed so 9 fits/fold do not flood the VM logs;
        no per-candidate ``.summary()`` is printed (unlike ``fit``).
        """

        import warnings

        seasonal_order = (0, 0, 0, 0)
        stationarity = self.cfg["models"]["SARIMAX"].get("enforce_stationarity", True)
        invertibility = self.cfg["models"]["SARIMAX"].get("enforce_invertibility", True)

        best = None  # (aic, order, fit)
        tried = 0
        failed = 0
        start_time = time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(p_max + 1):
                for q in range(q_max + 1):
                    order = (p, d, q)
                    tried += 1
                    try:
                        res = SARIMAX(endog=y_train, order=order, seasonal_order=seasonal_order,
                                      trend="n", enforce_stationarity=stationarity,
                                      enforce_invertibility=invertibility).fit(disp=False)
                        aic = float(res.aic)
                    except Exception:
                        failed += 1
                        continue
                    if not np.isfinite(aic):
                        failed += 1
                        continue
                    if best is None or aic < best[0]:
                        best = (aic, order, res)

        self.runtime_seconds = time() - start_time
        if best is None:
            raise RuntimeError(
                f"sarimax_aic: all {tried} candidate orders failed or returned "
                "non-finite AIC for this fold")

        self.last_order = best[1]
        self.last_seasonal_order = seasonal_order
        self.model_fit = best[2]
        print(f"[sarimax_aic] order={best[1]} AIC={best[0]:.2f} "
              f"({tried - failed}/{tried} candidates valid)")
        return best[2]



    def predict(self, model_fit: SARIMAXResults = None, y_test: pd.DataFrame = None,
                y_train: pd.DataFrame = None, include_intervals: bool = False) -> tuple:

        """
        Predicts the test data based on the fitted model.
        If y_test is provided and has an index, use get_prediction(start, end) to produce aligned preds and
        (optionally) conf_int.

        :param model_fit: The fitted model.
        :param y_test: The pandas Series containing the test data (actuals).
        :param y_train: The pandas Series used for training (used for MASE scale).
        :param include_intervals: If True try to return conf_int as well (not required).
        :return: (predictions Series, residuals Series, optional conf_int DataFrame)
        """

        eps = 1e-9

        if model_fit is None:
            model_fit = self.model_fit

        conf_int = None

        if y_test is None:
            raise ValueError("y_test must be provided to predict()")

        # Rolling 1-step-ahead within the test fold. For each day t we forecast 1
        # step from the current Kalman state, then update the state with the
        # realized observation via append(refit=False). Model parameters stay
        # frozen across the fold; refits happen between folds in walk-forward.
        # No silent fallback to multi-step forecasts — a failed append fails the
        # fold so we never reintroduce the classical/neural protocol asymmetry.
        y_test_vals = np.asarray(y_test, dtype=float).ravel()
        results = model_fit
        preds = []
        for pos, y_obs in enumerate(y_test_vals):
            f = results.forecast(steps=1)
            preds.append(float(np.asarray(f).ravel()[0]))
            if not np.isfinite(y_obs):
                # Fail the fold loudly. Silently skipping append would freeze
                # the state and misalign every subsequent prediction.
                raise ValueError(
                    f"Non-finite y_obs at test position {pos}: {y_obs}. "
                    "Refusing to advance the rolling state with NaN/inf."
                )
            results = results.append([float(y_obs)], refit=False)
        y_pred = pd.Series(preds, index=y_test.index)
        # include_intervals is only used by the legacy single-shot block; the
        # walk-forward path never requests intervals. Drop them here to keep the
        # rolling loop simple.
        conf_int = None

        # residuals (keep pandas index)
        residuals = y_test - y_pred
        residuals = pd.Series(residuals, index=y_test.index)

        # convert to numpy arrays to avoid pandas index-label alignment issues
        y_pred_arr = np.asarray(y_pred).astype(float)
        y_test_arr = np.asarray(y_test).astype(float)

        # defensive: ensure same length (trim if necessary)
        if y_pred_arr.shape[0] != y_test_arr.shape[0]:
            minlen = min(y_pred_arr.shape[0], y_test_arr.shape[0])
            y_pred_arr = y_pred_arr[:minlen]
            y_test_arr = y_test_arr[:minlen]

        # mask out non-finite pairs (NaN / inf) to avoid NaN metrics
        finite_mask = np.isfinite(y_pred_arr) & np.isfinite(y_test_arr)


        if finite_mask.sum() == 0:
            # nothing left to evaluate — produce safe None values and warn
            mse = None
            mae = None
            rmse = None
            directional_accuracy = None
            smape = None
            mase = None
            print("Warning: no finite prediction/actual pairs available for metric calculation.")

        else:
            yp = y_pred_arr[finite_mask]
            yt = y_test_arr[finite_mask]

            mse = float(np.mean((yt - yp) ** 2))
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(mse))
            directional_accuracy = float(np.mean(np.sign(yp) == np.sign(yt)))
            smape = float(100.0 * np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + eps)))

        # MASE scale based on training set if provided
        if y_train is not None:
            s = None

            if getattr(self, "last_seasonal_order", None) is not None:
                try:
                    s = int(self.last_seasonal_order[3])

                except Exception:
                    s = None

            if s is None:
                try:
                    s = int(self.cfg["models"]["SARIMAX"]["seasonal_order"][3])

                except Exception:
                    s = None

            if s and len(y_train) > s:
                scale_errors = np.abs(y_train.iloc[s:].values - y_train.iloc[:-s].values)

            else:
                if len(y_train) < 2:
                    scale_errors = np.array([eps])

                else:
                    scale_errors = np.abs(y_train.iloc[1:].values - y_train.iloc[:-1].values)

            scale = float(np.mean(scale_errors)) if scale_errors.size > 0 else float(eps)
            mase = float(mae / (scale + eps))

        else:
            mase = None

        print('Mean Squared Error: ', mse)
        print('Mean Absolut Error: ', mae)
        print('Root Mean Squared Error: ', rmse)
        print('Directional Acc.: ', directional_accuracy)
        print('SMAPE: ', smape)
        print('MASE: ', mase)

        # Per-fold metrics + logging are owned centrally by
        # main._log_folds_centrally / _fold_metrics. predict() keeps the
        # (y_pred, residuals, exp_id) tuple shape for the legacy
        # single-shot callers' signature.
        exp_id = None

        self.last_experiment_id = exp_id
        self.last_fold_id = None

        return y_pred, residuals, exp_id



    # ─── HISTORIC AND FUTURE FORECASTING ──────────────────────────────────────────────────────────────────────────────

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

        # SANITY CHECKS
        # check whether fitted model is available
        if self.model_fit is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        # determine experiment_id
        if experiment_id is None:
            experiment_id = getattr(self, "last_experiment_id", None)

            if experiment_id is None:
                raise ValueError("experiment_id not provided and no last_experiment_id found.")

        # check whether test dataset is available
        if y_test is None:
            raise ValueError("y_test dataset must be provided!")

        steps = len(y_test)

        # FORECAST
        try:
            forecast_res = self.model_fit.get_forecast(steps=steps)
            y_forecast = forecast_res.predicted_mean

            # reindex predicted_mean to y_test.index and fallback to forecast for missing entries
            if hasattr(y_forecast, "index"):
                y_forecast = y_forecast.reindex(y_test.index)

                if y_forecast.isna().any():
                    fallback = pd.Series(self.model_fit.forecast(len(y_test)), index=y_test.index)
                    y_forecast = y_forecast.fillna(fallback)

            else:
                y_forecast = pd.Series(y_forecast, index=y_test.index)

            conf_int = forecast_res.conf_int() if include_intervals else None

            if conf_int is not None and hasattr(conf_int, "index"):
                conf_int = conf_int.reindex(y_test.index)

        except Exception:
            # fallback: forecast and map to y_test index
            y_forecast = pd.Series(self.model_fit.forecast(steps), index=y_test.index)
            conf_int = None

        # Convert to arrays for metric computation
        preds = y_forecast.to_numpy(dtype=float)
        actuals = y_test.to_numpy(dtype=float)

        lowers = np.full_like(preds, np.nan)
        uppers = np.full_like(preds, np.nan)

        if conf_int is not None:
            lowers = conf_int.iloc[:, 0].to_numpy(dtype=float)
            uppers = conf_int.iloc[:, 1].to_numpy(dtype=float)

        # mask non-finite pairs
        finite_mask = np.isfinite(preds) & np.isfinite(actuals)

        # METRICS
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

        # coverage & mean prediction interval width
        interval_mask = np.isfinite(lowers) & np.isfinite(uppers) & finite_mask

        if interval_mask.sum() == 0:
            coverage = mean_pi_width = None

        else:
            L = lowers[interval_mask]
            U = uppers[interval_mask]
            A = actuals[interval_mask]
            coverage = float(np.mean((A >= L) & (A <= U)))
            mean_pi_width = float(np.mean(U - L))

        # MASE
        try:
            full_series = self.features["log_return"].copy()
            first_test_ts = y_test.index[0]

            if first_test_ts in full_series.index:
                train_series = full_series.loc[:first_test_ts].iloc[:-1]

            else:
                train_series = full_series.iloc[:-len(y_test)]

            if len(train_series) >= 2:

                s = None

                if getattr(self, "last_seasonal_order", None) is not None:
                    try:
                        s = int(self.last_seasonal_order[3])

                    except Exception:
                        s = None

                if s is None:
                    try:
                        s = int(self.cfg["models"]["SARIMAX"]["seasonal_order"][3])

                    except Exception:
                        s = None

                if s and len(train_series) > s:
                    scale_errors = np.abs(train_series.iloc[s:].values - train_series.iloc[:-s].values)

                else:
                    scale_errors = np.abs(train_series.iloc[1:].values - train_series.iloc[:-1].values)

                scale = float(np.mean(scale_errors)) if scale_errors.size > 0 else 1e-9
                mase = float(mae / (scale + 1e-9)) if mae is not None else None

            else:
                mase = None

        except Exception:
            mase = None

        # --- Build aggregated record ---
        row = {
            "experiment_id": experiment_id,
            "forecast_run_id": historic_forecast_run_id,
            "ticker": self.ticker,
            "model_type": "SARIMAX",
            "horizon": int(steps),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "smape": smape,
            "mase": mase,
            "directional_accuracy": directional_accuracy,
            "coverage": coverage,
            "mean_pi_width": mean_pi_width,
            "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id", None) is not None
                       else None,
            "notes": "SARIMAX run"
        }


        # ensure experiment_id stored for later convenience
        self.last_experiment_id = experiment_id

        # batch insert via logger
        inserted = self.logger.log_historic_forecast(row)



    def future_forecast(self, experiment_id: Optional[str] = None, steps: Optional[int] = None,
                        include_intervals: bool = True, o: Optional[List] = None, so: Optional[List] = None):

        """
        Produces out-of-sample forecasts for a given horizon (steps) and logs them to Supabase.

        :param experiment_id: Uuid from experiment_results (if None, forecast() will try to use self.last_experiment_id;
                              if that is missing, it will raise).
        :param steps: Integer horizon. If None, tries to read from cfg at cfg["models"]["SARIMAX"]["forecast_steps"].
        :param include_intervals: Whether to include prediction intervals (y_lower / y_upper).
        """

        future_forecast_run_id = str(uuid.uuid4())
        self.last_future_forecast_run_id = future_forecast_run_id

        endog = self.features["log_return"].copy()
        endog.name = "log_return"

        # determine order/seasonal_order to use: prefer explicit args, then last_* (from previous fit), then cfg
        order_to_use = tuple(o) if o is not None else (
                    getattr(self, "last_order", None) or tuple(self.cfg["models"]["SARIMAX"]["order"]))

        seasonal_to_use = tuple(so) if so is not None else (getattr(self, "last_seasonal_order", None) or tuple(
            self.cfg["models"]["SARIMAX"]["seasonal_order"]))

        # re-fit on the full available series using the chosen combo
        future_model_fit = self.fit(endog, order_to_use, seasonal_to_use)

        # determine steps
        if steps is None:
            try:
                steps = int(self.cfg["models"]["SARIMAX"].get("forecast_steps", 10))

            except Exception:
                steps = 10

        if experiment_id is None:
            experiment_id = getattr(self, "last_experiment_id", None)

            if experiment_id is None:
                raise ValueError("experiment_id not provided and no last_experiment_id found.")

        # get forecast object
        forecast_res = future_model_fit.get_forecast(steps=steps)
        y_forecast = forecast_res.predicted_mean

        conf_int = None

        if include_intervals:
            try:
                conf_int = forecast_res.conf_int()

            except Exception:
                conf_int = None

        rows = []

        # For robustness, use y_pred.index when plausible, otherwise build an iso list.
        if hasattr(y_forecast, "index") and not isinstance(y_forecast.index, pd.RangeIndex):
            idx = y_forecast.index

        else:
            # fallback: create simple next-day iso strings
            last_date = pd.Timestamp(endog.index[-1])
            idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')

        for i, ts in enumerate(idx):
            ts_str = pd.Timestamp(ts).to_pydatetime().isoformat() if isinstance(ts, (pd.Timestamp, datetime))\
                                                                  else str(ts)

            forecast_val = float(y_forecast.iloc[i]) if hasattr(y_forecast, "iloc") else float(y_forecast[i])
            lower = float(conf_int.iloc[i, 0]) if (conf_int is not None) else None
            upper = float(conf_int.iloc[i, 1]) if (conf_int is not None) else None
            pi_width = (upper - lower) if (lower is not None and upper is not None) else None

            row = {
                "experiment_id": experiment_id,
                "forecast_run_id": future_forecast_run_id,
                "ticker": self.ticker,
                "model_type": "SARIMAX",
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
                "fold_id": int(getattr(self, "last_fold_id", None)) if getattr(self, "last_fold_id", None) is not None
                           else None,
                "horizon": int(steps),
                "notes": "SARIMAX run"
            }
            rows.append(row)

        # ensure experiment_id stored for future convenience
        self.last_experiment_id = experiment_id

        # batch insert via logger
        inserted = self.logger.log_future_forecast(rows)



    # ─── VISUALISING RESULTS ──────────────────────────────────────────────────────────────────────────────

    def results_vis(self, predictions: pd.Series = None, residuals: pd.Series = None):

        """
        Visualising the results via the matplotlib.pyplot library.

        :param predictions:
        :param residuals:
        """

        plt.figure(figsize=(10, 4))
        plt.plot(residuals)
        plt.axhline(0, linestyle='--', color='k')
        plt.title('Residuals from SARIMAX Model', fontsize=20)
        plt.ylabel('Error', fontsize=16)
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/SARIMA/Residuals/residuals.png")

        plt.figure(figsize=(10, 4))
        plt.plot(self.features["log_return"])
        plt.plot(predictions)
        plt.legend(('Data', 'Predictions'), fontsize=12)
        plt.title(f'{self.ticker} — log returns and predictions', fontsize=16)
        plt.ylabel('Log. Returns', fontsize=12)
        plt.savefig("/Users/juliusschmidt/Desktop/BachelorThesisExperimentVis/SARIMA/Predictions/predictions.png")