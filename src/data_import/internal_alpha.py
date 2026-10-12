import requests
import pandas as pd
from typing import Dict

class InternalAlphaVantageDataWrapper:

    """
    Minimal wrapper for Alpha Vantage API fundamental (financial statements) endpoints:
        - (Quarterly or yearly) Income statements
        - (Quarterly or yearly) Balance sheets
        - (Quarterly or yearly) Cashflow statements
        - (Quarterly or yearly) Earnings reports

    All data is returned as JSON and parsed into pandas DataFrames.
    """

    # Base URL for using the Alpha Vantage API
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, alpha_api_key: str):

        """
        Constructor / Initialising the 'InternalAlphaVantageDataWrapper' object with Alpha Vantage API key.

        :param alpha_api_key: Your Alpha Vantage API key.
        """

        self.alpha_api_key = alpha_api_key


    def fetch(self, function: str, ticker: str) -> Dict:

        """
        Fetches internal data of a publicly listed company via the Alpha Vantage API.

        :param function: The Alpha Vantage function name, e.g. 'INCOME_STATEMENT'.
        :param symbol: The stock ticker symbol, e.g. 'AAPL'.
        :return: Parsed JSON response as a Python dict.
        :raises: HTTPError if the request fails.
        """

        params = {"function": function, "symbol": ticker.upper(), "apikey": self.alpha_api_key}

        resp = requests.get(self.BASE_URL, params=params)
        resp.raise_for_status()

        return resp.json()


    def json_to_df(self, data: Dict, key: str) -> pd.DataFrame:

        """
        Converts a JSON block (dict) from Alpha Vantage into a pandas DataFrame.

        This expects `data[key]` to be a list of dicts, each containing
        a 'fiscalDateEnding' field representing the report date.

        :param data: The full JSON response from Alpha Vantage.
        :param key: The top-level key for the report list, e.g. 'annualReports' or 'quarterlyReports'.
        :return: A DataFrame indexed by report date (fiscalDateEnding).
        :raises: ValueError if `data[key]` is missing or empty.
        """

        # if user requested quarterlyReports but it's not there, fallback to annualReports
        if key == "quarterlyReports" and "quarterlyReports" not in data:
            if "annualReports" in data:
                records = data["annualReports"]
            else:
                # still nothing, bail out
                return pd.DataFrame()
        else:
            records = data.get(key, [])

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Parse and set the date index
        df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])
        df.set_index("fiscalDateEnding", inplace=True)

        # Coerce all other columns to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_index()


    # ─── GETTER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def get_income_statement(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:

        """
        Fetches the income statement for a given symbol.

        :param symbol: The stock ticker symbol, e.g. 'AAPL'.
        :param quarterly: If False, fetches the annual statement via 'INCOME_STATEMENT' and 'annualReports';
                          if True, fetches the quarterly statement via 'QUARTERLY_INCOME_STATEMENT'
                          and 'quarterlyReports'.
        :return: A DataFrame of income statement line items, indexed by fiscal date ending.
        """

        function = "INCOME_STATEMENT"
        key = "annualReports" if not quarterly else "quarterlyReports"
        json_data = self.fetch(function, ticker)

        return self.json_to_df(json_data, key)


    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:

        """
        Fetches the balance sheet for a given symbol.

        :param symbol: The stock ticker symbol, e.g. 'AAPL'.
        :param quarterly: If False, fetches the annual statement via 'INCOME_STATEMENT' and 'annualReports';
                          if True, fetches the quarterly statement via 'QUARTERLY_INCOME_STATEMENT'
                          and 'quarterlyReports'.
        :return: A DataFrame of balance sheet line items, indexed by fiscal date ending.
        """

        function = "BALANCE_SHEET"
        key = "annualReports" if not quarterly else "quarterlyReports"
        json_data = self.fetch(function, ticker)

        return self.json_to_df(json_data, key)


    def get_cash_flow(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:

        """
        Fetches the cash flow statement for a given symbol.
        Alpha Vantage only supports the annual 'CASH_FLOW'; quarterly cash flows are returned under the same key.


        :param symbol: The stock ticker symbol, e.g. 'AAPL'.
        :param quarterly: If False, fetches the annual statement via 'INCOME_STATEMENT' and 'annualReports';
                          if True, fetches the quarterly statement via 'QUARTERLY_INCOME_STATEMENT'
                          and 'quarterlyReports'.
        :return: A DataFrame of cash flow line items, indexed by fiscal date ending.
        """

        function = "CASH_FLOW"
        key = "annualReports" if not quarterly else "quarterlyReports"
        json_data = self.fetch(function, ticker)

        return self.json_to_df(json_data, key)


    def get_earnings(self, ticker: str, quarterly: bool = False) -> pd.DataFrame:

        """
        Fetches earnings report data for a given symbol.

        :param ticker: The stock ticker symbol, e.g. 'AAPL'.
        :param quarterly: If False, fetches the annual statement via 'INCOME_STATEMENT' and 'annualReports';
                          if True, fetches the quarterly statement via 'QUARTERLY_INCOME_STATEMENT'
                          and 'quarterlyReports'.
        :return: A DataFrame of earnings reports indexed by fiscalDateEnding, with columns like 'reportedEPS', etc.
        """

        function = "EARNINGS"
        key = "annualEarnings" if not quarterly else "quarterlyEarnings"
        json_data = self.fetch(function, ticker)

        return self.json_to_df(json_data, key)