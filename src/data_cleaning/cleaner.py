import pandas as pd
from typing import Dict, Any

class DataCleaner:

    """
    Takes raw data dicts and:
        - Exports them to Excel for manual inspection.
        - Provides methods for cleaning the data inside the dicts.

    All methods return cleaned pandas DataFrames or dicts.
    """

    def __init__(self, cfg: Dict[str, Any]):

        """
        Constructor / Initialising the 'DataCleaner' object with the config.yaml.

        :param cfg: Config.yaml contents.
        """

        self.cfg = cfg


    def export_to_excel(self, internal: Dict[str, Any],
                            external: Dict[str, Any], path: str = "dump.xlsx"):

        """
        Writes all internal & external tables (and dicts) into one Excel file.
            - DataFrames: Sheet named "int_<key>" or "ext_<key>".
            - Dicts: Converted to 2-column DataFrame and written similarly.
            - Timezones are stripped from all datetime columns to avoid Excel errors.

        :param internal: Internal data fetched by the Yahoo Finance API.
        :param external: External data fetched by both FRED API and Yahoo Finance API.
        :param path: Path to the data dump as .xlsx.
        """

        # Helper alias
        tz_strip = self.remove_timezones

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:

            # Internal data
            for name, data in internal.items():
                sheet_name = f"int_{name}"[:31] # Excel max sheet name length.

                if isinstance(data, pd.DataFrame):
                    df = tz_strip(data.copy())
                    df.to_excel(writer, sheet_name=sheet_name)

                elif isinstance(data, dict):
                    # Converts dict to DataFrame with two columns: key, value.
                    df = pd.DataFrame(list(data.items()), columns=["key", "value"])
                    df = tz_strip(df)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                else:
                    # Skip other types (e.g. lists).
                    continue

            # External FRED data
            fred_dict = external.get("fred", {})

            for name, series in fred_dict.items():
                sheet_name = f"ext_fred_{name}"[:31] # Excel max sheet name length.

                # Make into DataFrame for Excel
                df = series = series.to_frame(name)
                df = tz_strip(df)
                df.to_excel(writer, sheet_name=sheet_name)

            # External Yahoo Finance indices (and ETFs).
            for tkr, df in external.get("indices", {}).items():
                sheet_name = f"ext_idx_{tkr}"[:31] # Excel max sheet name length

                if isinstance(df, pd.DataFrame):
                    tz_strip(df.copy()).to_excel(writer, sheet_name=sheet_name)

                else:
                    continue



    # ─── GENERIC CLEANING METHODS ─────────────────────────────────────────────────────────────────────────────────────

    def remove_timezones(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Strips timezone information from all tz-aware datetime columns and index.

        :param df: The DataFrame being stripped off any timezone information.
        :return: The DataFrame without any timezone information.
        """

        # Columns
        for col in df.select_dtypes(include=["datetimetz"]).columns:
            df[col] = df[col].dt.tz_convert(None).dt.tz_localize(None)

        # Index
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None).tz_localize(None)

        return df



    def to_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Converts the index of a DataFrame to a datetime index.

        :param df: DataFrame of which the index is being set to datetime.
        :return: Returns the converted DataFrame.
        """

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df



    # ─── PRICE / INDEX CLEANING METHODS ───────────────────────────────────────────────────────────────────────────────

    def clean_price(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Ensures that OHLCV DataFrames contain no duplicates, is business-day indexed and has no missing values.

        :param df: The raw OHLCV DataFrame being cleaned.
        :return: The cleaned new OHLCV DataFrame which was cleaned.
        """

        df = df.drop_duplicates()
        df = self.to_datetime_index(df)

        # Reindex to business days, forward-fill
        df = df.asfreq("B").ffill()
        df = self.remove_timezones(df)

        return df



    # ─── MACRO SERIES CLEANING METHODS─────────────────────────────────────────────────────────────────────────────────

    def clean_fred_series(self, s: pd.Series) -> pd.Series:

        """
        Cleans one FRED Series of macro data by filling/interpolating it by its inferred frequency.

        :param s: The raw FRED Series being cleaned.
        :return: The newly cleaned FRED Series.
        """

        # Drop leading/trailing NaNs
        s = s.dropna()

        # Reindex to a regular grid ONLY when pandas can confidently infer the
        # native frequency. The old code did `freq=s.index.inferred_freq or "ME"`,
        # which is a silent-data-loss trap: a MONTH-START series whose latest month
        # is not yet released (e.g. CPI, UNRATE, CIVPART, Core CPI — published with a
        # ~1-month lag) loses its inferred freq after dropna, falls back to "ME"
        # (month-END), and reindexes onto 30th/31st dates that DON'T overlap the
        # 1st-of-month data → every value becomes NaN. The same fallback silently
        # DOWNSAMPLED daily series (treasury yields, WTI, GVZ) to monthly. So: only
        # reindex when the freq is real; otherwise interpolate on the native index
        # (downstream feature building reindexes onto business days + ffills anyway,
        # so internal gaps are still handled). See tests/test_clean_fred_series.py.
        inferred = s.index.inferred_freq
        if inferred:
            idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=inferred)
            s = s.reindex(idx)

        # Interpolate numeric
        s = s.interpolate(method="time").ffill().bfill()

        return s



    # ─── FINANCIAL STATEMENTS CLEANING METHODS ────────────────────────────────────────────────────────────────────────

    def clean_financial_statement(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Cleans financial statements fetched by the FRED API. For both quarterly and annual statements, is being ensured
        that values are numeric, missing values are being filled with zeros, and it's transposed so that index=period,
        and columns=line items.

        :param df: The raw financial statement in form of a pandas DatFrame which is being cleaned.
        :return: The cleaned financial statement in form of a pandas DataFrame.
        """

        df = df.copy()

        # Transpose if needed so rows are timestamps
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.T

        df.index = pd.to_datetime(df.index)

        # Coerce numeric
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df = df.sort_index()

        return df



    def clean_income_statement(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Cleans a raw income statement.

        :param df: Raw earnings report (as pandas DataFrame) being cleaned.
        :return: Cleaned earnings report as pandas DataFrame.
        """

        return self.clean_financial_statement(df)



    def clean_balance_sheet(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Cleans a raw balance sheet.

        :param df: Raw earnings report (as pandas DataFrame) being cleaned.
        :return: Cleaned earnings report as pandas DataFrame.
        """

        return self.clean_financial_statement(df)



    def clean_cashflow(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Cleans a raw cashflow report.

        :param df: Raw earnings report (as pandas DataFrame) being cleaned.
        :return: Cleaned earnings report as pandas DataFrame.
        """

        return self.clean_financial_statement(df)



    def clean_earnings(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Cleans a raw earnings report.

        :param df: Raw earnings report (as pandas DataFrame) being cleaned.
        :return: Cleaned earnings report as pandas DataFrame.
        """

        return self.clean_financial_statement(df)



    # ─── METADATA CLEANING METHODS ────────────────────────────────────────────────────────────────────────────────────

    def clean_dict(self, d: Dict[str, Any], keys_to_keep: list = None) -> Dict[str, Any]:

        """
        Filters and standardises a metadata dict, e.g. extract industry or sector.

        :param d: Metadata dict which is being filtered and standardised.
        :param keys_to_keep: Metadata which is being filtered for.
        :return: Returns the filtered and standardised dict.
        """

        if keys_to_keep:
            return {k: d.get(k) for k in keys_to_keep}

        return d



    def clean_calendar(self, d: Dict[str, Any]) -> Dict[str, pd.Timestamp]:

        """
        Converts calendar dates to Timestamp.

        :param d: Dict containing the stocks calendar data.
        :return: Converted dict containing the stocks calendar data.
        """

        data = {}

        for k, v in d.items():

            try:
                data[k] = pd.to_datetime()

            except Exception:
                data[k] = v

        return data



    # ─── TOP‐LEVEL CLEANING METHODS ───────────────────────────────────────────────────────────────────────────────────

    def clean_internal(self, raw_int: Dict[str, Any]) -> Dict[str, Any]:

        """
        Top-level method for cleaning all raw, internal data of the stock being analysed and forecasted.
        Including the following data:
            - Price data
            - Info data
            - Calendar data
            - Yearly income statement data
            - Quarterly income statement data
            - Yearly balance sheet data
            - Quarterly balance sheet data
            - Yearly cashflow report data
            - Quarterly cashflow report data
            - Yearly earnings report data
            - Earnings dates data

        :param raw_int: The raw, internal data (Yahoo Finance API) of the stock being analysed and forecasted.
        :return: Returns the cleaned internal data as a dict "ci".
        """

        # Dict storing proper and cleaned data, for now empty
        ci = {}

        # Assigning the proper and cleaned data to the final dict
        ci["prices"] = self.clean_price(raw_int["prices"])

        ci["info"] = self.clean_dict(raw_int["info"], keys_to_keep=["sector", "industry"])

        ci["income_q"] = self.clean_income_statement(raw_int["income_q"])

        ci["balance_sheet_q"] = self.clean_balance_sheet(raw_int["balance_sheet_q"])

        ci["cashflow_q"] = self.clean_cashflow(raw_int["cashflow_q"])

        # ci["earnings_q"] = self.clean_earnings(raw_int["earnings_q"])

        # Skipping analyst recommendations, sustainability, and holders, due to missing time series connection.

        return ci



    def clean_external(self, raw_ext: Dict[str, Any]) -> Dict[str, Any]:

        """
        Top-level method for cleaning all raw, external data being used to forecast the chosen stocks.
        Including the following data:
            - U.S. Gross Domestic Product
            - U.S. Consumer Price Index
            - U.S. Core CPI
            - U.S. Personal Consumption Expenditures
            - U.S. Core PCE
            - U.S. Unemployment Rate
            - U.S. Nonfarm Payrolls
            - U.S. Labor Force Participation
            - U.S. Average Hourly Earnings
            - U.S. Weekly Initial Jobless Claims
            - U.S. Federal Funds Rate
            - U.S. Treasury Yields - 1Y
            - U.S. Treasury Yields - 2Y
            - U.S. Treasury Yields - 5Y
            - U.S. Treasury Yields - 10Y
            - U.S. Treasury Yields - 30Y
            - U.S. Corporate Bonds Spread
            - U.S. 10Y–2Y Treasury Spread
            - U.S. Industrial Production Index
            - U.S. Capacity Utilization (Total)
            - U.S. Real Manufacturing & Trade Sales (Inflation-Adjusted)
            - U.S. Producer Price Index
            - U.S. Broad Retail Sales
            - U.S. Corporate Profits After Tax
            - U.S. Consumer Sentiment
            - U.S. Personal Savings Rate
            - U.S. Housing Starts
            - U.S. Building Permits
            - U.S. Existing Home Sales
            - U.S. Case–Shiller Home Price Index
            - U.S. M2 Money Stock
            - U.S. Total Commercial Banks Credit
            - U.S. Credit Card Delinquencies
            - U.S. Chicago Fed NFCI
            - U.S. Trade Balance
            - U.S. Imports
            - U.S. Exports
            - U.S. Crude Oil (WTI)
            - U.S. Gold Volatility (CBOE GVZ)
            - S&P500 Index
            - Nasdaq Index
            - Dow Jones Index
            - DAX Index
            - EuroStoxx 50 Index
            - Gold ETF
            - iShares MSCI World ETF

        :param raw_ext: All raw, external data (FRED and Yahoo Finance API) being used to forecast the chosen stocks.
        :return: Returns the cleaned external data as a dict "ce".
        """

        # Dict storing proper and cleaned data, for now empty
        ce = {}

        # Macro dict of FRED API series, all series cleaned
        ce["fred"] = {name: self.clean_fred_series(s) for name, s in raw_ext["fred"].items()}

        # Indices, commodities, and ETFs dict of DataFrames, all cleaned
        ce["indices"] = {t: self.clean_price(df) for t, df in raw_ext["indices"].items()}

        return ce