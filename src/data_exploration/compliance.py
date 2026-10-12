import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from typing import Dict, Any, List, Optional
import warnings

class DataExplorationCompliance():

    """
    Quick compliance check of the data being used to fit and forecast.
    """

    def __init__(self, features: pd.DataFrame):

        """
        Constructor / Initialising a "DataExplorationVisualisation" object.

        :param features: The pandas DataFrame of features being visualised.
        """

        self.features = features.copy()



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def is_constant(self, feature, tol=1e-12) -> bool:

        """
        Check if a series is (almost) constant, ignoring NaNs.

        :param feature: The feature column being analysed.
        :param tol:
        :return:
        """

        f = pd.Series(feature).dropna()

        if f.empty:
            return True

        return np.all(np.abs(f - f.iloc[0]) < tol)



    # ─── ADFULLER x KPSS TESTING AND REPORTING ────────────────────────────────────────────────────────────────────────

    def adfuller_test(self, feature) -> Dict[str, Any]:

        """
        Checking whether a feature is stationary via the ADFuller test.

        :param feature: The feature column being analysed.
        :return: Result array of ADFuller test.
        """

        if len(feature) < 10 or self.is_constant(feature):
            return {
                "adf_stat": np.nan,
                "adf_p": np.nan,
                "constant": True,
                "n": len(feature)
            }

        adf = adfuller(feature)

        return {
            "adf_stat": adf[0],
            "adf_p": adf[1],
            "constant": False,
            "n": len(feature)
        }



    def kpss_test(self, feature) -> Dict[str, Any]:

        """
        Checking whether a feature is stationary via the ADFuller test.

        :param feature: The feature column being analysed.
        :return: Result array of the KPSS test.
        """

        if len(feature) < 10 or self.is_constant(feature):
            return {
                "kpss_stat": np.nan,
                "kpss_p": np.nan,
                "constant": True,
                "n": len(feature)
            }

        kpss_stat, kpss_p, _, _ = kpss(feature, regression="c", nlags="auto")

        return {
            "kpss_stat": kpss_stat,
            "kpss_p": kpss_p,
            "constant": False,
            "n": len(feature)
        }



    def stationarity_report(self):

        """
        Runs ADFuller and KPSS test on each column. Both prints the loggers into console + returns a summary DataFrame.

        :return: The loggers of both the ADFuller and the KPSS tests on each column as a pandas DataFrame.
        """

        alpha = 0.05

        cols = self.features.columns
        rows = []

        for c in cols:
            feature = pd.to_numeric(self.features[c], errors="coerce")

            adf = self.adfuller_test(feature)
            kpss = self.kpss_test(feature)

            # decisions
            adf_stationary = False if np.isnan(adf["adf_p"]) else (adf["adf_p"] < alpha)
            kpss_stationary = False if np.isnan(kpss["kpss_p"]) else (kpss["kpss_p"] >= alpha)

            # if constant, mark as stationary by definition
            if adf.get("constant") or kpss.get("constant"):
                verdict = "constant (stationary)"

            else:
                adf_stationary = adf["adf_p"] < alpha if not np.isnan(adf["adf_p"]) else False
                kpss_stationary = kpss["kpss_p"] >= alpha if not np.isnan(kpss["kpss_p"]) else False

                if adf_stationary and kpss_stationary:
                    verdict = "stationary"

                elif (not adf_stationary) and (not kpss_stationary):
                    verdict = "non-stationary"

                else:
                    verdict = "inconclusive"

            rows.append({
                "column": c,
                "adf_p": adf.get("adf_p"),
                "kpss_p": kpss.get("kpss_p"),
                "adf_stationary": adf_stationary,
                "kpss_stationary": kpss_stationary,
                "verdict": verdict,
                "n_obs": int(adf.get("n", kpss.get("n", 0)))
            })

        return pd.DataFrame(rows).set_index("column")



    # ─── ACF x PACF TESTING AND REPORTING ─────────────────────────────────────────────────────────────────────────────

    def acf_pacf_test(self, feature, nlags: int = 40, alpha: float = 0.05) -> Dict[str, Any]:

        """
        Computes ACF and PACF for a single series and detects significant lags.

        :param feature: The feature column being analysed.
        :param nlags:
        :param alpha:
        :return: Returns a dict with:
                  - acf: numpy array (length nlags+1)
                  - acf_confint: (nlags+1, 2) array or None
                  - pacf: numpy array (length nlags+1)
                  - pacf_confint: (nlags+1, 2) array or None
                  - significant_acf_lags: list of ints (lags where acf is significant)
                  - significant_pacf_lags: list of ints (lags where pacf is significant)
                  - n_obs: int, number of non-na observations
                  - note: string (if computation skipped)
        """

        feature = pd.Series(feature.copy()).dropna()
        n = len(feature)

        result: Dict[str, Any] = {"acf": None, "acf_confint": None, "pacf": None, "pacf_confint": None,
                                  "significant_acf_lags": [], "significant_pacf_lags": [],
                                  "n_obs": n, "note": None}

        # safety checks
        if n < 10:
            result["note"] = "too_short"

            return result

        if self.is_constant(feature):
            result["note"] = "constant_series"

            # return zeros for acf/pacf to keep shape consistent
            result["acf"] = np.zeros(nlags + 1)
            result["pacf"] = np.zeros(nlags + 1)

            return result

        # compute ACF
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acf_result = acf(feature.values, nlags=nlags, alpha=alpha, fft=False)

            # acf_result may be tuple (acf, confint) depending on statsmodels version
            if isinstance(acf_result, tuple) and len(acf_result) == 2:
                acf_vals, acf_conf = acf_result

            else:
                acf_vals = np.asarray(acf_result)
                acf_conf = None

        except Exception:
            # fallback without confint
            acf_vals = acf(feature.values, nlags=nlags, fft=False)
            acf_conf = None

        # compute PACF
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pacf_res = pacf(feature.values, nlags=nlags, alpha=alpha, method="ywm")

            # acf_result may be tuple (acf, confint) depending on statsmodels version
            if isinstance(pacf_res, tuple) and len(pacf_res) == 2:
                pacf_vals, pacf_conf = pacf_res

            else:
                pacf_vals = np.asarray(pacf_res)
                pacf_conf = None

        except Exception:
            # fallback without confint
            pacf_vals = pacf(feature.values, nlags=nlags, method="ywm")
            pacf_conf = None

        result["acf"] = np.asarray(acf_vals)
        result["acf_confint"] = np.asarray(acf_conf) if acf_conf is not None else None
        result["pacf"] = np.asarray(pacf_vals)
        result["pacf_confint"] = np.asarray(pacf_conf) if pacf_conf is not None else None

        # identify significant lags
        sig_acf: List[int] = []
        sig_pacf: List[int] = []

        # if confint available, use it; otherwise use approx. Bartlett's SE:  ±1.96/sqrt(n)
        if result["acf_confint"] is not None:
            for lag in range(1, min(len(result["acf"]), nlags + 1)):
                lo, hi = result["acf_confint"][lag]
                val = result["acf"][lag]

                if not (lo <= val <= hi):
                    sig_acf.append(lag)

        else:
            thresh = 1.96 / np.sqrt(n)

            for lag in range(1, min(len(result["acf"]), nlags + 1)):
                if abs(result["acf"][lag]) > thresh:
                    sig_acf.append(lag)

        if result["pacf_confint"] is not None:
            for lag in range(1, min(len(result["pacf"]), nlags + 1)):
                lo, hi = result["pacf_confint"][lag]
                val = result["pacf"][lag]

                if not (lo <= val <= hi):
                    sig_pacf.append(lag)
        else:
            thresh = 1.96 / np.sqrt(n)

            for lag in range(1, min(len(result["pacf"]), nlags + 1)):
                if abs(result["pacf"][lag]) > thresh:
                    sig_pacf.append(lag)

        result["significant_acf_lags"] = sig_acf
        result["significant_pacf_lags"] = sig_pacf

        return result



    def acf_pacf_report(self, columns: Optional[List[str]] = None, nlags: int = 40, alpha: float = 0.05,
                        sig_lag_frac_for_slow_decay: float = 0.2) -> pd.DataFrame:

        """
        Runs ACF/PACF analysis on columns and returns a summary DataFrame.

        :param columns: What features to analyse. Default is None, all columns will be analysed.
        :param nlags:
        :param alpha:
        :param sig_lag_frac_for_slow_decay:
        :return: The summary contains:
                  - n_obs: number of observations
                  - n_sig_acf_lags: how many acf lags are significant (lag>=1)
                  - n_sig_pacf_lags: how many pacf lags are significant (lag>=1)
                  - acf_lag1: acf at lag 1
                  - pacf_lag1: pacf at lag 1
                  - heuristic: short human-readable pattern (e.g. 'slow_acf_decay', 'AR(p)', 'MA(q)', 'likely_stationary')
                  - suggestion: what to try next ('difference', 'log-diff', 'keep as-is', 'fit AR(p)')
        """

        cols = columns or list(self.features.columns)
        rows = []

        for c in cols:
            s = pd.to_numeric(self.features[c], errors="coerce")
            r = self.acf_pacf_test(s, nlags=nlags, alpha=alpha)

            if r.get("note") == "too_short":
                rows.append({
                    "column": c,
                    "n_obs": r["n_obs"],
                    "n_sig_acf_lags": np.nan,
                    "n_sig_pacf_lags": np.nan,
                    "acf_lag1": np.nan,
                    "pacf_lag1": np.nan,
                    "heuristic": "too_short",
                    "suggestion": "skip / not enough data"
                })
                continue

            if r.get("note") == "constant_series":
                rows.append({
                    "column": c,
                    "n_obs": r["n_obs"],
                    "n_sig_acf_lags": 0,
                    "n_sig_pacf_lags": 0,
                    "acf_lag1": 0.0,
                    "pacf_lag1": 0.0,
                    "heuristic": "constant",
                    "suggestion": "constant (stationary) — drop or keep"
                })
                continue

            acf_arr = r["acf"]
            pacf_arr = r["pacf"]
            n_sig_acf = len(r["significant_acf_lags"])
            n_sig_pacf = len(r["significant_pacf_lags"])

            # heuristics
            acf_lag1 = float(acf_arr[1]) if len(acf_arr) > 1 else np.nan
            pacf_lag1 = float(pacf_arr[1]) if len(pacf_arr) > 1 else np.nan

            # slow decay detection: many significant acf lags OR acf at high lag still > threshold
            slow_decay = False

            if n_sig_acf >= int(sig_lag_frac_for_slow_decay * nlags):
                slow_decay = True

            # also if the acf at nlags/2 is relatively large:
            midlag = min(len(acf_arr) - 1, max(2, nlags // 2))

            if abs(acf_arr[midlag]) > 0.2:
                slow_decay = True

            # assemble heuristic & suggestion
            heuristic = "unknown"
            suggestion = "inspect visually"

            # very strong lag1 autocorrelation often indicates high persistence / unit root
            if abs(acf_lag1) > 0.9 or slow_decay:
                heuristic = "slow_acf_decay"
                suggestion = "likely non-stationary -> try differencing (or log-diff for levels)"

            else:
                # if pacf has a single strong spike at lag p and pacf drops after => AR(p)
                if n_sig_pacf > 0 and (len(r["significant_pacf_lags"]) <= 3):
                    p = r["significant_pacf_lags"][0]
                    heuristic = f"AR_like_p={p}"
                    suggestion = f"consider AR({p}) or include lag {p} as feature"

                # if acf has isolated significant lag(s) at low lags => MA(q)
                elif n_sig_acf > 0 and (max(r["significant_acf_lags"]) <= 5):
                    q = max(r["significant_acf_lags"])
                    heuristic = f"MA_like_q={q}"
                    suggestion = f"consider MA({q})"

                else:
                    heuristic = "likely_stationary"
                    suggestion = "keep as-is / no differencing required"

            rows.append({
                "column": c,
                "n_obs": r["n_obs"],
                "n_sig_acf_lags": n_sig_acf,
                "n_sig_pacf_lags": n_sig_pacf,
                "acf_lag1": acf_lag1,
                "pacf_lag1": pacf_lag1,
                "heuristic": heuristic,
                "suggestion": suggestion
            })

        return pd.DataFrame(rows).set_index("column")



    # ─── TRANSFORMING DATA II / CREATING STATIONARITY ─────────────────────────────────────────────────────────────────

    def make_price_data_stationary(self) -> pd.DataFrame:

        """
        Transforming OCHL values to log. differences closing prices.

        :param df: The pandas DataFrame containing all features being transformed.
        :return: Transformed pandas DataFrame.
        """

        columns = ["open", "high", "low"]

        df_new = self.features.copy()

        for c in columns:
            df_new[c] = np.log(df_new[c]).diff()

        # drop "close" column -> since when log. diff. loggers in "log. return"
        df_new = df_new.drop('close', axis=1)

        return df_new



    def make_indices_data_stationary(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Transforming closing prices of all indices and ETFs to log. differences closing prices.

        :param df: The pandas DataFrame containing all features being transformed.
        :return: Transformed pandas DataFrame.
        """

        columns = ["gspc", "ndx", "dji", "gdaxi", "stoxx50e", "gld", "urth", "acwi"]

        df_new = df.copy()

        # Some index columns may legitimately be absent — e.g. the per-ticker
        # leakage guard drops ^GSPC ("gspc") when forecasting SPY. Only transform
        # the columns that are actually present.
        for c in columns:
            if c in df_new.columns:
                df_new[c] = np.log(df_new[c]).diff()

        return df_new



    def stationarize_reporting_series(self, df: pd.DataFrame, prefixes: tuple) -> pd.DataFrame:

        """
        Generic helper to convert sparsely reported series (e.g. quarterly/weekly/monthly)
        into stationarized features.

        For every column in `df` whose name starts with any of `prefixes` we create 4 new columns:
          - {col}_pctchg           : sparse percent-change on reporting days
          - {col}_pctchg_ffill     : forward-filled pct-change to daily frequency (no invented history)
          - {col}_logdiff          : sparse log-diff where previous and current > 0
          - {col}_logdiff_ffill    : forward-filled log-diff to daily frequency (no invented history)

        :param df: The pandas Dataframe being transformed to be stationary.
        :param prefixes: The prefixes of the columns being transformed to be stationary.
        :return: The transformed pandas DataFrame.
        """

        df_new = df.copy()
        idx = pd.to_datetime(df_new.index)

        # store new columns here instead of inserting repeatedly
        new_data = {}

        for col in list(df_new.columns):
            if not col.startswith(prefixes):
                continue

            # numeric copy with proper datetime index
            s = pd.to_numeric(df_new[col].copy(), errors="coerce")
            s.index = pd.to_datetime(s.index, errors="coerce")

            pct_sparse = pd.Series(index=idx, dtype=float)
            log_sparse = pd.Series(index=idx, dtype=float)

            if s.dropna().empty:
                new_data[f"{col}_pctchg"] = pct_sparse
                new_data[f"{col}_pctchg_ffill"] = pct_sparse.copy()
                new_data[f"{col}_logdiff"] = log_sparse
                new_data[f"{col}_logdiff_ffill"] = log_sparse.copy()

                continue

            prev = s.shift(1)
            report_mask = s.notna() & ((s != prev) | prev.isna())
            reports = s.where(report_mask).dropna()

            if reports.empty:
                new_data[f"{col}_pctchg"] = pct_sparse
                new_data[f"{col}_pctchg_ffill"] = pct_sparse.copy()
                new_data[f"{col}_logdiff"] = log_sparse
                new_data[f"{col}_logdiff_ffill"] = log_sparse.copy()

                continue

            # percent change on reporting days.
            # Guard the zero-denominator case: when a prior reported value is
            # 0 (or the series crosses through 0), pct_change yields +/-inf
            # (undefined "0 -> X" growth). Map those to NaN so the fillna(0.0)
            # at the end of this method treats them as "no change", mirroring
            # how the log-diff branch below already drops non-positive entries.
            # Without this, inf leaks into {col}_pctchg(_ffill) and the sklearn
            # scaler in the neural models raises "Input X contains infinity".
            reports_pct = reports.pct_change().replace([np.inf, -np.inf], np.nan)

            if reports_pct.size >= 1 and pd.isna(reports_pct.iat[0]):
                reports_pct.iat[0] = 0.0

            prev_reports = reports.shift(1)
            valid_log_mask = (prev_reports > 0) & (reports > 0)

            reports_logdiff = pd.Series(index=reports.index, dtype=float)

            if valid_log_mask.any():
                reports_logdiff.loc[valid_log_mask] = np.log(reports.loc[valid_log_mask]) - np.log(
                    prev_reports.loc[valid_log_mask])

            pct_sparse.loc[reports_pct.index] = reports_pct.values
            log_sparse.loc[reports_logdiff.index] = reports_logdiff.values

            pct_ffill = pct_sparse.ffill()
            log_ffill = log_sparse.ffill()

            first_report = reports.index[0]
            pct_ffill.loc[pct_ffill.index < first_report] = np.nan
            log_ffill.loc[log_ffill.index < first_report] = np.nan

            # collect instead of insert
            new_data[f"{col}_pctchg"] = pct_sparse
            new_data[f"{col}_pctchg_ffill"] = pct_ffill
            new_data[f"{col}_logdiff"] = log_sparse
            new_data[f"{col}_logdiff_ffill"] = log_ffill

        # attach all new columns at once — avoids fragmentation
        if new_data:
            new_df = pd.DataFrame(new_data, index=idx)
            df_new = pd.concat([df_new, new_df], axis=1)

            # optional: replace NaNs with 0.0 only for new cols
            df_new[list(new_data.keys())] = df_new[list(new_data.keys())].fillna(0.0)

        return df_new



    def make_financials_data_stationary(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Apply the generic stationarizer to quarterly financial prefixes.
        """

        prefixes = ("income_q", "balance_sheet_q", "cashflow_q")

        return self.stationarize_reporting_series(df, prefixes=prefixes)



    def make_macro_data_stationary(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Apply the generic stationarizer to macro series prefixed 'us_'.
        """

        prefixes = ("us_",)

        return self.stationarize_reporting_series(df, prefixes=prefixes)



    def create_stationarity(self, condition: str = "transformed", warmup_discard: int = 20,
                            skip_exog_stationarization: bool = False) -> pd.DataFrame:

        """
        Apply stationarity transforms and discard the NaN warmup period.

        Condition 1 ("transformed"): full log-diff and pct-change pipeline.
        Condition 2 ("raw"): no transforms — raw feature levels passed through.
        Both conditions discard the same warmup_discard rows so walk-forward
        fold indices are aligned across conditions.

        :param condition: "transformed" | "raw"
        :param warmup_discard: rows to drop from the start (default 20)
        :param skip_exog_stationarization: run #2b only. When True, do NOT
            stationarize the exogenous (index + macro) columns here, because they
            were ALREADY stationarized upstream by the exog_stationarize transform
            table (features/stationarize.py). Re-running make_indices_data_stationary
            would log-diff the already-log-returned indices -> log(negative/0) ->
            -inf (the smoke-gate launch-blocker), and make_macro_data_stationary
            would build reporting-series derivatives from already-differenced macro.
            The INTERNAL price + financial-statement transforms still run (they are
            not in the exog table). One layer owns each column's stationarity.
        :return: Transformed (or raw) pandas DataFrame.
        """

        if condition == "raw":
            return self.features.iloc[warmup_discard:]

        df_price_stat = self.make_price_data_stationary()
        if skip_exog_stationarization:
            # Exog already stationarized by the table -> skip make_indices + make_macro.
            df_final = self.make_financials_data_stationary(df_price_stat)
        else:
            df_idcs_stat = self.make_indices_data_stationary(df_price_stat)
            df_fund_stat = self.make_financials_data_stationary(df_idcs_stat)
            df_final = self.make_macro_data_stationary(df_fund_stat)

        return df_final.iloc[warmup_discard:]