import yfinance as yf
import pandas as pd
from typing import List

class InternalYahooFinanceDataWrapper:

    """
    Wrapper for internal data sources:
        - Yahoo Finance (stocks)

    All methods return raw pandas Series, DataFrames, Lists or Dicts.
    """

    def __init__(self, ticker: str, period: str = "5y", interval: str = "1d"):

        """
        Constructor / Initialising the 'InternalDataWrapper' object with ticker symbol, period, and interval.

        :param ticker: Ticker symbol of specific stock, e.g. "AAPL".
        :param period: Period for which data should be fetched, e.g. "5y" or "2mo".
        :param interval: Interval for which data should be fetched, e.g. "1d" or "1mo".
        """

        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.period = period
        self.interval = interval


    # ─── PRICE DATA / Open, High, Low, Close, Volume, Dividends, Splits ───────────────────────────────────────────────

    def get_price_data(self) -> pd.DataFrame:

        """
        Fetches historical raw price data (OHLCVDS), using 'Date' as index.
        """

        df = self.stock.history(period=self.period, interval=self.interval)
        df.index.name = "Date"

        return df


    # ─── META DATA ────────────────────────────────────────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:

        """
        Fetches historical raw stock info data and store in a dict (due to multiple 'dtypes').
        """

        return self.stock.info

    def get_calendar(self) -> dict:

        """
        Fetches historical calendar data and store in a dict (due to multiple 'dtypes').
        """

        return self.stock.calendar


    # ─── FUNDAMENTAL DATA ─────────────────────────────────────────────────────────────────────────────────────────────

    def get_income_statement(self) -> pd.DataFrame:

        """
        Fetches historical raw income statement report data, yearly.
        """

        return self.stock.financials

    def get_quarterly_income_statement(self) -> pd.DataFrame:

        """
        Fetches historical raw income statement report data, quarterly.
        """

        return self.stock.quarterly_financials

    def get_balance_sheet(self) -> pd.DataFrame:

        """
        Fetches historical raw balance sheet report data, yearly.
        """

        return self.stock.balance_sheet

    def get_quarterly_balance_sheet(self) -> pd.DataFrame:

        """
        Fetches historical raw balance sheet report data, quarterly.
        """

        return self.stock.quarterly_balance_sheet

    def get_cashflow(self) -> pd.DataFrame:

        """
        Fetches historical raw cashflow report data, yearly.
        """

        return self.stock.cashflow

    def get_quarterly_cashflow(self) -> pd.DataFrame:

        """
        Fetches historical raw cashflow report data, quarterly.
        """

        return self.stock.quarterly_cashflow


    # ─── EARNINGS DATA ────────────────────────────────────────────────────────────────────────────────────────────────

    def get_earnings(self) -> pd.DataFrame:

        """
        Fetches historical raw earnings report data, yearly.
        """

        return self.stock.income_stmt

    def get_quarterly_earnings(self) -> pd.DataFrame:

        """
        Fetches historical raw earnings report data, quarterly.
        """

        return self.stock.quarterly_earnings

    def get_earnings_dates(self) -> pd.DataFrame:

        """
        Fetches historical raw earnings dates.
        """

        return self.stock.earnings_dates


    # ─── ANALYSTS ANALYSIS DATA ───────────────────────────────────────────────────────────────────────────────────────

    def get_recommendations(self) -> pd.DataFrame:

        """
        Fetches analysts recommendations.
        """

        return self.stock.recommendations

    def get_analysis(self) -> pd.DataFrame:

        """
        Fetches analysts stock analysis.
        """

        return self.stock._analysis


    # ─── ESG / SUSTAINABILITY ─────────────────────────────────────────────────────────────────────────────────────────

    def get_sustainability(self) -> pd.DataFrame:

        """
        Fetches sustainability score(s).
        """

        return self.stock.sustainability


    # ─── HOLDERS ──────────────────────────────────────────────────────────────────────────────────────────────────────

    def get_major_holders(self) -> pd.DataFrame:

        """
        Fetches current major holders.
        """

        return self.stock.major_holders

    def get_institutional_holders(self) -> pd.DataFrame:

        """
        Fetches current major institutional holders.
        """

        return self.stock.institutional_holders

    def get_mutualfund_holders(self) -> pd.DataFrame:

        """
        Fetches current major mutualfund holders.
        """

        return self.stock.mutualfund_holders


    # ─── OPTIONS ──────────────────────────────────────────────────────────────────────────────────────────────────────

    def get_options_dates(self) -> List[str]:

        """
        Fetches historical options dates.
        """

        return self.stock.options

    def get_option_chain(self, date= str):

       """
       Fetches historical option chain.
       """

       return self.stock.option_chain(date)