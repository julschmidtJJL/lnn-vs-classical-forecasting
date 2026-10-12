import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable



class FeatureBuilder:

    """
    Constructs a unified feature DataFrame, given cleaned internal and external data.
    """

    def __init__(self, cfg: Dict[str, Any]):

        """
        Constructor / Initialising the 'FeatureBuilder' object with config.yaml file.

        :param cfg: Your config.yaml file.
        """

        self.cfg = cfg

        # Flatten feature-metadata sections for easy lookup later
        fm = cfg.get("features", {})

        self.feature_meta: Dict[str, Dict[str, Any]] = {}

        def sanitize(name: str) -> str:
            s = name.lower().replace(".", "")
            s = re.sub(r"[^a-z0-9]+", "_", s)
            return s.strip("_")

        # ─── internal ──────────────────────────────
        int_meta = fm.get("internal", {})

        # 1) price & info & technical indicators
        for subgroup in ("price", "info", "technical_indicators"):
            for feat_name, meta in int_meta.get(subgroup, {}).items():
                key = sanitize(feat_name)
                self.feature_meta[key] = meta

        # 2) financial statements (quarterly)
        fin = int_meta.get("financial_statement", {})

        for stmt_name, lines in fin.items():
            suffix = "" if stmt_name.endswith("_q") else "_q"

            for line_name, meta in lines.items():
                key = sanitize(f"{stmt_name}{suffix}_{line_name}")
                self.feature_meta[key] = meta

        # ─── macro ─────────────────────────────────
        for feat_name, meta in fm.get("macro", {}).items():
            key = sanitize(feat_name)
            self.feature_meta[key] = meta

        # ─── index ─────────────────────────────────
        for feat_name, meta in fm.get("index", {}).items():
            key = sanitize(feat_name)
            self.feature_meta[key] = meta

        # Build a registry of transform functions
        self.transforms: Dict[str, Callable[..., pd.Series]] = {

            "none": lambda s, **kw: s,
            "pct_change": lambda s, periods, **kw: s.pct_change(periods=periods) * 100,
            "log_return": lambda s, **kw: np.log(s).diff(),
            # First difference. The stationarizing transform for series that can be
            # zero or negative (rates, yields, spreads) where log_return/pct_change
            # are undefined or unstable. Run #2b stationarized-exog arm.
            "diff": lambda s, periods=1, **kw: s.diff(periods=periods),

        }



    # ─── HELPER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def sanitize(self, name: str) -> str:

        """
        Lowercases, replaces any sequence of non-alphanumeric characters with a single underscore,
        then strips leading/trailing underscores.

        :param name: The name being sanitized.
        :return: The sanitized name.
        """

        # lowercase, then drop literal dots
        s = name.lower().replace(".", "")

        # replace any run of characters that is NOT a–z or 0–9 with underscore
        s = re.sub(r"[^a-z0-9]+", "_", s)

        # drop leading/trailing underscores
        return s.strip("_")



    def ensure_datetime_index(self, obj):

        """
        Ensures a pandas Series/DataFrame has a proper DatetimeIndex (timezone-naive), sorted ascending. Returns a copy.

        :param obj: The Series/DataFrame which is being formatted to proper DatetimeIndex.
        :return: The newly formatted Series/DataFrame.
        """

        # Convert index to datetime if not already
        if not isinstance(obj.index, pd.DatetimeIndex):
            obj.index = pd.to_datetime(obj.index, errors="coerce")

        # Drop NaT indices just in case
        obj = obj.loc[~obj.index.isna()]

        # Sort by index and ensure unique
        obj = obj[~obj.index.duplicated(keep="last")].sort_index()

        return obj



    # ─── INTERNAL (PRICES & FINANCIALS) FEATURES ──────────────────────────────────────────────────────────────────────

    def build_internal_price(self, price_df: pd.DataFrame) -> pd.DataFrame:

        """
        Compute basic price features from cleaned OHLCV DataFrame.

        :param price_df: Cleaned price DataFrame with columns:
                         'Open','High','Low','Close','Volume','Dividends', and 'Stock Splits'.
        :return: DataFrame of internal price features:
                         - open, high, low, close (USD)
                         - log_return (log-return)
                         - volume (shares)
                         - dividends (USD)
                         - stock_splits (factor)
        """

        df = pd.DataFrame(index=price_df.index)

        # Prices
        df["open"] = price_df["Open"]
        df["high"] = price_df["High"]
        df["low"] = price_df["Low"]
        df["close"] = price_df["Close"]

        # Log returns
        df["log_return"] = np.log(price_df["Close"]).diff()

        # Volume, dividends, and splits
        df["volume"] = price_df["Volume"]
        df["dividends"] = price_df["Dividends"]
        df["stock_splits"] = price_df["Stock Splits"]

        return df



    def build_financial_features(self, clean_internal: Dict[str, Any]) -> pd.DataFrame:

        """
        Takes quarterly income, BS, CF and earnings, forward-fills to business days, prefixes their columns,
        and applys any declared transformations.

        :param clean_internal: Cleaned internal data containing a company's financial statements.
        :return: The pandas DataFrame containing all financial statement features.
        """

        data = {}

        for section in ("income_q", "balance_sheet_q", "cashflow_q", "earnings_q"):
            df_q = clean_internal.get(section)

            if df_q is None:
                continue

            if not isinstance(df_q, pd.DataFrame):
                continue

            # make sure index is datetime and columns numeric
            df_q = self.ensure_datetime_index(df_q)

            # coerce all columns to numeric (invalid -> NaN)
            df_q = df_q.apply(pd.to_numeric, errors="coerce")

            # resample to business days and forward-fill
            daily = df_q.resample("B").ffill()

            for col in daily.columns:
                s = daily[col]

                if s.dropna().empty:
                    continue

                key = f"{section}_{self.sanitize(col)}"
                meta = self.feature_meta.get(key, {})

                transform = meta.get("transform", "none")
                params = meta.get("params", {})

                # ensure numeric series before applying transform
                s_numeric = pd.to_numeric(s, errors="coerce")

                try:
                    out_series = self.transforms[transform](s_numeric, **params)

                except Exception as e:
                    out_series = s_numeric

                data[key] = out_series

        if not data:
            return pd.DataFrame()

        result = pd.DataFrame(data)

        # final cleanup: only numeric columns (keeps dtype stable)
        result = result.select_dtypes(include="number")

        # align index to business days and ensure ffilled
        result = result.asfreq("B").ffill()

        return result



    # ─── EXTERNAL / MACRO FEATURES ────────────────────────────────────────────────────────────────────────────────────

    def build_macro_features(self, clean_external: Dict[str, Any]) -> pd.DataFrame:

        """
        Compute macroeconomic features from cleaned FRED series.

        :param clean_external: Dict containing cleaned external data, where
                               external["fred"] is a dict of pandas Series.
        :return: DataFrame of daily‐frequency macro features (reindexed to business days):
                 one column per config key, with transforms applied (e.g. none, pct_change_12m).
        """

        fred_obj = clean_external.get("fred")

        if fred_obj is None:
            return pd.DataFrame()

        # If it's a DataFrame, iterate columns; if dict, iterate items.
        if isinstance(fred_obj, pd.DataFrame):
            items = fred_obj.items()

        elif isinstance(fred_obj, dict):
            items = fred_obj.items()

        else:
            return pd.DataFrame()

        data = {}

        for long_name, series in items:
            # sanitize the series (coerce to Series, ensure datetime index)
            if isinstance(series, pd.DataFrame):
                # if a frame slipped in, try to pick a sensible column (e.g. first)

                if series.shape[1] >= 1:
                    series = series.iloc[:, 0]

                else:
                    continue

            series = self.ensure_datetime_index(series)
            series = pd.to_numeric(series, errors="coerce")

            key = self.sanitize(long_name)
            meta = self.feature_meta.get(key, {})

            transform = meta.get("transform", "none")
            params = meta.get("params", {})

            try:
                data[key] = self.transforms[transform](series, **params)

            except Exception as e:
                data[key] = series

            non_null = data[key].dropna().shape[0]

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data).asfreq("B").ffill()

        return df



    def build_index_features(self, clean_external: Dict[str, Any]) -> pd.DataFrame:

        """
        Compute index/ETF features from cleaned Yahoo Finance data.

        :param clean_external: Dict containing cleaned external data, where
                               clean_external["indices"] is a dict of DataFrames.
        :return: DataFrame of business‐day index features:
                 one column per config key, using Close prices and transforms.
        """

        idx = clean_external.get("indices", {}) or {}

        data = {}

        for ticker, df_idx in idx.items():
            key = self.sanitize(ticker)
            meta = self.feature_meta.get(key, {})

            transform = meta.get("transform", "none")
            params = meta.get("params", {})

            # pick Close safely
            if isinstance(df_idx, pd.DataFrame):
                if "Close" in df_idx.columns:
                    series = df_idx["Close"].copy()

                else:
                    # fallback to first numeric column
                    num_cols = df_idx.select_dtypes(include="number").columns

                    if len(num_cols) == 0:
                        continue

                    series = df_idx[num_cols[0]].copy()

            elif isinstance(df_idx, pd.Series):
                series = df_idx.copy()

            else:
                continue

            # sanitize timestamp index and coerce numeric
            series = self.ensure_datetime_index(series)
            series = pd.to_numeric(series, errors="coerce")

            # normalize timestamps to midnight (important to align different timezones)
            if isinstance(series.index, pd.DatetimeIndex):
                series.index = series.index.normalize()

            # apply transform
            try:
                data[key] = self.transforms[transform](series, **params)

            except Exception as e:
                data[key] = series

        if not data:
            return pd.DataFrame()

        # assemble, ensure business-day frequency and forward-fill
        df = pd.DataFrame(data)

        # ensure the assembled df has a DatetimeIndex and normalized dates
        df.index = pd.to_datetime(df.index, errors="coerce")

        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.normalize()

        df = df.asfreq("B").ffill()

        return df



    # ─── TECHNICAL INDICATORS FEATURES ────────────────────────────────────────────────────────────────────────────────

    def build_technical_features(self, price_df: pd.DataFrame) -> pd.DataFrame:

        """
        Compute standard technical indicators from cleaned OHLCV DataFrame.

        :param price_df: Cleaned price DataFrame with columns:
                         'Open','High','Low','Close','Volume','Dividends','Stock Splits'.
        :return: DataFrame containing one column per indicator, indexed by Date.
        """

        df = pd.DataFrame(index=price_df.index)

        close = price_df["Close"]
        high = price_df["High"]
        low = price_df["Low"]
        vol = price_df["Volume"]

        meta = self.cfg["features"]["internal"]["technical_indicators"]

        # 1) Simple Moving Average
        w = meta["simple_moving_average"]["params"]["window"]

        df["simple_moving_average"] = close.rolling(window=w).mean()

        # 2) Exponential Moving Average
        w = meta["exponential_moving_average"]["params"]["window"]

        df["exponential_moving_average"] = close.ewm(span=w, adjust=False).mean()

        # 3) MACD
        p = meta["moving_average_convergence_divergence"]["params"]
        fast = close.ewm(span=p["fast_window"], adjust=False).mean()
        slow = close.ewm(span=p["slow_window"], adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=p["signal_window"], adjust=False).mean()

        df["moving_average_convergence_divergence"] = macd - signal

        # 4) RSI
        w = meta["relative_strength_index"]["params"]["window"]
        delta = close.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.ewm(com=w - 1, adjust=False).mean()
        roll_down = down.ewm(com=w - 1, adjust=False).mean()
        rs = roll_up / roll_down

        df["relative_strength_index"] = 100 - (100 / (1 + rs))

        # 5) ATR
        w = meta["average_true_range"]["params"]["window"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        df["average_true_range"] = tr.rolling(window=w).mean()

        # 6) OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()

        df["on_balance_volume"] = obv

        # 7) CMF
        w = meta["chaikin_money_flow"]["params"]["window"]
        mf_mult = ((close - low) - (high - close)) / (high - low)
        mf_vol = mf_mult * vol

        df["chaikin_money_flow"] = mf_vol.rolling(window=w).sum() / vol.rolling(window=w).sum()

        # 8) Bollinger Bands
        w, n = meta["upper_bollinger_band"]["params"]["window"], meta["upper_bollinger_band"]["params"]["std_dev"]
        sma = close.rolling(window=w).mean()
        std = close.rolling(window=w).std()

        df["upper_bollinger_band"] = sma + n * std
        df["lower_bollinger_band"] = sma - n * std

        # 9) Stochastic Oscillator
        p = meta["stochastic_oscillator"]["params"]
        low_min = low.rolling(window=p["window"]).min()
        high_max = high.rolling(window=p["window"]).max()
        k_pct = 100 * (close - low_min) / (high_max - low_min)

        df["stochastic_oscillator"] = k_pct.rolling(window=p["smooth_window"]).mean()

        # 10) Keltner Channel
        kp = meta["upper_keltner_channel"]["params"]
        ema = close.ewm(span=kp["ema_window"], adjust=False).mean()
        atr = df["average_true_range"]

        df["upper_keltner_channel"] = ema + kp["multiplier"] * atr.rolling(window=kp["atr_window"]).mean()
        df["lower_keltner_channel"] = ema - kp["multiplier"] * atr.rolling(window=kp["atr_window"]).mean()

        # 11) Donchian Channel
        w = meta["upper_donchian_channel"]["params"]["window"]

        df["upper_donchian_channel"] = high.rolling(window=w).max()
        df["lower_donchian_channel"] = low.rolling(window=w).min()

        # 12) Rate of Change
        w = meta["rate_of_change"]["params"]["window"]

        df["rate_of_change"] = close.pct_change(periods=w) * 100

        # 13) Williams %R
        w = meta["williams_r"]["params"]["window"]
        lowest_low = low.rolling(window=w).min()
        highest_high = high.rolling(window=w).max()

        df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low)

        # 14) ADX (simplified)
        w = meta["average_directional_index"]["params"]["window"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()

        tr = tr  # from ATR calculation above
        atr = tr.rolling(window=w).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / w, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / w, adjust=False).mean() / atr)

        df["average_directional_index"] = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

        return df



    # ─── MAIN ASSEMBLER ───────────────────────────────────────────────────────────────────────────────────────────────

    def build_features(self, clean_internal: Dict[str, Any], clean_external: Dict[str, Any],
                       condition: str = "transformed", macro_lag: int = 1,
                       ticker: str = None) -> pd.DataFrame:

        """
        Assemble all feature groups into a single DataFrame.

        :param clean_internal: Dict of cleaned internal data (must include "prices").
        :param clean_external: Dict of cleaned external data ("fred" and "indices").
        :return: DataFrame indexed by business‐day Date with columns:
                 - internal price features (open, high, low, close, log_return, etc.)
                 - internal financial statement features (net profit, ebit, etc.)
                 - macro features (one column per macro config key)
                 - index features (one column per index config key)
        """

        # 1) Internal price features (this is the canonical index we align to)
        price_features = self.build_internal_price(clean_internal["prices"])
        price_features = self.ensure_datetime_index(price_features)  # <-- ensures price index is clean
        price_index = price_features.index

        # 2) Internal financial statement features (quarterly -> daily)
        finance_features = self.build_financial_features(clean_internal)

        if finance_features is None:
            finance_features = pd.DataFrame()

        # 3) Macro features (FRED)
        macro_features = self.build_macro_features(clean_external)

        if macro_features is None:
            macro_features = pd.DataFrame()

        # 4) Index features (Yahoo indices)
        index_features = self.build_index_features(clean_external)

        # Per-ticker exclusions: drop index columns that would leak the target.
        # Example: when forecasting SPY, drop ^GSPC (sanitized to "gspc") because
        # it's a near-direct proxy of the same underlying basket. Configured under
        # cfg["data"]["feature_exclusions"]: {SPY: ["^GSPC"]}.
        if ticker is not None and index_features is not None and not index_features.empty:
            exclusions_cfg = (self.cfg.get("data", {}) or {}).get("feature_exclusions", {}) or {}
            excluded = exclusions_cfg.get(ticker, [])
            if excluded:
                drop_cols = [self.sanitize(name) for name in excluded]
                drop_present = [c for c in drop_cols if c in index_features.columns]
                if drop_present:
                    print(f"[FeatureBuilder] Dropping leakage-risk index columns for "
                          f"{ticker}: {drop_present}")
                    index_features = index_features.drop(columns=drop_present)

        # 5) Technical indicator features — skipped in raw condition (D1: strict raw)
        if condition == "transformed":
            technical_indicator_features = self.build_technical_features(clean_internal["prices"])
        else:
            technical_indicator_features = pd.DataFrame()

        # --- Helper to align any group to the price index and forward-fill ---
        def align_to_price(df: pd.DataFrame, name: str, allow_bfill_earlier=False) -> pd.DataFrame:

            if df is None or df.empty:
                return pd.DataFrame(index=price_index)

                # make sure df has a clean DatetimeIndex
            df = self.ensure_datetime_index(df)

            # normalize both indices to date-only (midnight) so we align by calendar day
            df_date = df.copy()
            df_date.index = df_date.index.normalize()

            # price dates normalized:
            price_dates_norm = price_index.normalize()

            # Reindex to the normalized price dates and forward-fill
            re = df_date.reindex(price_dates_norm).ffill()

            # Optionally backfill earlier price dates with the first available observation
            if allow_bfill_earlier:
                re = re.bfill()

            # Coerce columns to numeric where possible
            for c in re.columns:
                re[c] = pd.to_numeric(re[c], errors="coerce")

            # Now restore the original price timestamp index (so all groups share exact same final index)
            re.index = price_index

            non_null_cols = re.dropna(how="all", axis=1).shape[1]

            return re

        macro_features = align_to_price(macro_features, "macro")

        index_features = align_to_price(index_features, "indices", allow_bfill_earlier=True)

        technical_indicator_features = align_to_price(technical_indicator_features, "technical")

        finance_features = align_to_price(finance_features, "finance")

        # Track macro and index column names before merge so we can shift them
        macro_cols = list(macro_features.columns)
        index_cols = list(index_features.columns)

        # 6) Merging all on the canonical business-day index (price)
        df = price_features.join(macro_features, how="left") \
            .join(index_features, how="left") \
            .join(technical_indicator_features, how="left") \
            .join(finance_features, how="left")

        # 7) Publication-delay correction: shift all FRED and index columns by macro_lag.
        # Applied in both conditions to avoid look-ahead bias: a macro release on day T is
        # reported after market close, so models predicting T's return must use T-1's values.
        if macro_lag > 0:
            cols_to_lag = [c for c in (macro_cols + index_cols) if c in df.columns]
            if cols_to_lag:
                df[cols_to_lag] = df[cols_to_lag].shift(macro_lag)

        # Cutting off the first rows: tech indicator rolling warmup + financial statement alignment.
        df = df.iloc[33:]

        return df