import pandas as pd
import yfinance as yf
from fredapi import Fred
from typing import Dict, List, Optional

class ExternalDataWrapper:

    """
    Wrapper for external data sources:
        - FRED (macroeconomic time series)
        - Yahoo Finance (indices, commodities, etc.)

    All methods return raw pandas Series or DataFrames.
    """

    def __init__(self, fred_api_key: str, tickers: Optional[List[str]] = None):

        """
        Constructor / Initialising the 'ExternalDataWrapper' object with FRED API key.

        :param fred_api_key: Your FRED API key.
        :param tickers: Optional list of index/ETF tickers to preload via yfinance.
        """

        self.fred = Fred(api_key=fred_api_key)

        self.tickers = tickers or []

        # Preload yfinance Ticker objects for indices
        self._yf_tickers: Dict[str, yf.Ticker] = {
            tkr.upper(): yf.Ticker(tkr.upper())
            for tkr in self.tickers
        }


    # ─── FRED BULK FETCH ──────────────────────────────────────────────────────────────────────────────────────────────

    def get_fred_multiple(self, series_map: Dict[str, str], start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, pd.Series]:

        """
        Fetch multiple FRED series in one dict.

        :param series_map: Mapping {column_name: fred_series_id (sid)}.
        :param start_date: Start date of the timeframe being fetched.
        :param end_date: End date of the timeframe being fetched.
        :return: Dict indexed by date, columns are the chosen names.
        """

        data: Dict[str, pd.Series] = {}

        for name, sid in series_map.items():
            s = self.fred.get_series(sid, observation_start=start_date, observation_end=end_date)
            s.name = name
            data[name] = s

        return data


    # ─── YAHOO FINANCE BULK FETCH ─────────────────────────────────────────────────────────────────────────────────────

    def get_multiple_indices(self, tickers: List[str], period: str = "5y",
                             interval: str = "1d") -> Dict[str, pd.DataFrame]:

        """
        Fetch OHLCV histories for multiple indices, commodities, and ETFs.

        :param tickers: List of ticker strings.
        :param period: Period for which data should be fetched, e.g. "5y" or "2mo".
        :param interval: Interval for which data should be fetched, e.g. "1d" or "1mo".
        :return: Dict mapping symbol → DataFrame.
        """

        data: Dict[str, pd.DataFrame] = {}

        for tkr in tickers:
            yf_tkr = self._yf_tickers.get(tkr.upper(), yf.Ticker(tkr.upper()))

            # cache if new
            self._yf_tickers[tkr.upper()] = yf_tkr
            df = yf_tkr.history(period=period, interval=interval)
            df.index.name = "Date"
            data[tkr.upper()] = df

        return data