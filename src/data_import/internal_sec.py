import requests
import pandas as pd
from typing import Optional


class InternalSECWrapper:

    """
    Minimal wrapper for SEC API (financial statements / XBRL filings) endpoints:
        - (Quarterly) Income statements
        - (Quarterly) Balance sheets
        - (Quarterly) Cashflow statements
        - (Quarterly) Earnings reports
    """

    # Public mapping of tickers → CIK
    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

    # Companyfacts endpoint for XBRL facts
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"



    def __init__(self, user_agent: str, cfg: dict):

        """
        Constructor / Initialising the 'InternalSECWrapper' object with SEC user agent.

        :param user_agent: A string identifying the application per SEC policy.
        """

        self.headers = {"User-Agent": user_agent}
        self.cfg = cfg
        self.ticker_map = None



    def load_ticker_map(self) -> dict:

        """
        Loads (and caches) the mapping of ticker symbols → 10-digit CIKs.

        :return: Dict mapping uppercase ticker → zero-padded CIK string.
        """

        if self.ticker_map is None:
            r = requests.get(self.TICKER_MAP_URL, headers=self.headers)

            r.raise_for_status()
            raw = r.json()

            # building the ticker → CIK (10-digit with leading zeros)
            self.ticker_map = {
                v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in raw.values()
            }

        return self.ticker_map



    def get_cik(self, ticker: str) -> str:

        """
        Retrieves the SEC CIK for a given ticker symbol.

        :param ticker: The stock ticker symbol, e.g. 'AAPL'.
        :return: The corresponding 10-digit CIK string.
        :raises KeyError: If ticker not found in SEC’s mapping.
        """

        tm = self.load_ticker_map()
        t = ticker.upper()

        if t not in tm:
            raise KeyError(f"Ticker '{ticker}' not found in SEC mapping")

        return tm[t]



    def get_facts(self, cik: str) -> dict:

        """
        Fetches the US-GAAP XBRL facts payload for a company.

        :param cik: 10-digit CIK string.
        :return: The nested dict under ["facts"]["us-gaap"] from the JSON response.
        """

        url = self.FACTS_URL.format(cik=cik)

        r = requests.get(url, headers=self.headers)
        r.raise_for_status()

        return r.json()["facts"]["us-gaap"]



    def extract_series(self, facts: dict, concept: str) -> pd.Series:

        """
        From the US-GAAP facts dict pull out one concept series (USD unit by default), returning a pd.Series indexed by
        the SEC filing date (the date the figure became public), values in USD.

        :param facts: The dict returned by `get_facts()`.
        :param concept: XBRL tag, e.g. "us-gaap_NetIncomeLoss".
        :return: A pandas Series indexed by filing date, values in USD. Empty Series if the concept is missing.
        """

        entry = facts.get(concept)

        if not entry:
            return pd.Series(dtype=float)

        # pick USD if available
        units = entry["units"]
        records = units.get("USD") or next(iter(units.values()))

        df = pd.DataFrame(records)

        # Index by the SEC FILING date ("filed"), NOT the fiscal period-end
        # ("end"). A quarter's figures only become public on the day the
        # 10-Q/10-K is filed (~34 days after period-end for AAPL). Indexing by
        # "end" and forward-filling would let models see the numbers ~34 days
        # before they were released — lookahead leakage. "filed" lives in the
        # same record; fall back to "end" only if it is somehow absent.
        date_col = "filed" if "filed" in df.columns else "end"
        df = df.rename(columns={date_col: "date", "val": "value"})
        df["date"] = pd.to_datetime(df["date"])

        if date_col == "filed" and "end" in df.columns:
            # A fiscal period can be reported more than once (original filing
            # plus later restatements / comparatives). Keep the ORIGINAL filing
            # — the earliest "filed" date carries the value the market actually
            # had at the time; restated numbers were not known then.
            df["end"] = pd.to_datetime(df["end"])
            df = df.sort_values("date").drop_duplicates(subset="end", keep="first")
            # If one filing first-reports several periods on the same day, keep
            # the most recent period for that filing date.
            df = df.sort_values("end").drop_duplicates(subset="date", keep="last")
        else:
            # Fallback path (no "filed"): preserve prior behaviour.
            df = df.drop_duplicates(subset="date", keep="last")

        # set index (filing date) and extract the series
        return df.set_index("date")["value"].sort_index()



    def wide(self, ticker: str, concepts: list[str]) -> pd.DataFrame:

        """
        Build a wide DataFrame for the given ticker & list of GAAP concept names.

        :param ticker: Stock ticker, e.g. "AAPL".
        :param concepts: List of XBRL tags to include.
        :return: DataFrame indexed by quarter-end dates, columns are sanitized concept names (e.g. "netincomeloss"),
                 containing the last ~20 quarters.
        """

        cik = self.get_cik(ticker)
        facts = self.get_facts(cik)

        data = {}

        for concept in concepts:
            series = self.extract_series(facts, concept)

            # sanitize column name: drop 'us-gaap_' & lowercase
            col = concept.lower().replace("us-gaap_", "").replace("-", "_")
            data[col] = series

        df = pd.DataFrame(data).sort_index()

        # keep only last ~20 quarters (~5 years)
        return df.iloc[-20:]



    def snake_to_camel(self, snake: str) -> str:

        """
        Convert snake_case -> CamelCase (Basic heuristic).
        """

        return "".join(part.title() for part in snake.split("_"))



    def find_best_fact_key(self, facts: dict, snake: str) -> Optional[str]:

        """
        Try to find the best-matching facts key for a snake key by inspecting labels/
        descriptions. Returns the facts dict key or None.
        """

        tokens = [t for t in snake.replace("_", " ").split() if t]
        tokens = [t.lower() for t in tokens]

        best_key = None
        best_score = 0

        for key, entry in facts.items():
            label = (entry.get("label") or "").lower()

            desc = (entry.get("description") or "").lower()
            score = 0

            # token matches in label/description give +1 each
            for t in tokens:
                if t in label:
                    score += 3  # prefer label hits

                elif t in desc:
                    score += 1

            # small bonus if the snake's camel candidate is a substring of the fact key
            camel = self.snake_to_camel(snake).lower()

            if camel and camel in key.lower():
                score += 2

            # prefer shorter label distance (heuristic)
            if score > best_score:
                best_score = score
                best_key = key

        # require at least one token match to accept
        return best_key if best_score > 0 else None



    def extract_and_name(self, facts: dict, fact_key: str, col_name: str) -> Optional[pd.Series]:

        """
        Extract series for a fact_key and return the series named by col_name (or None if empty).
        """

        try:
            s = self.extract_series(facts, fact_key)

        except Exception:
            return None

        if s.empty:
            return None

        s.name = col_name

        return s



    def build_statement(self, ticker: str, section: str) -> pd.DataFrame:

        """
        Generic helper to pull the named concept-list in config under
        data.internal.financial_statement_concepts.<section>, attempt each configured XBRL concept
        (primary + alternatives) for each desired output column, fetch the first non-empty series,
        and assemble a wide df.

        The config structure expected is:
          data.internal.financial_statement_concepts:
            <section>:
              <output_col_name>:
                concept: "<PrimaryXBRLTag>"
                alternatives:
                  - "<AltTag1>"
                  - "<AltTag2>"
                force_column: true|false   # optional, default false

        :param ticker: str ticker
        :param section: Which section (income_q, balance_sheet_q, cashflow_q)
        :return: DataFrame with snake_case column names (one per configured output key)
        """

        cfg_section = self.cfg["features"]["internal"].get("financial_statement", {}).get(section, {})

        if not cfg_section:
            return pd.DataFrame()

        cik = self.get_cik(ticker)
        facts = self.get_facts(cik)

        data = {}

        for out_col, entry in cfg_section.items():
            # entry expected: {"concept": "...", "alternatives": [...], "force_column": bool}
            if not isinstance(entry, dict):
                # backwards compatibility: if user put a string or list directly, treat as list of concepts
                if isinstance(entry, str):
                    candidates = [entry]

                elif isinstance(entry, list):
                    candidates = entry

                else:
                    continue
            else:
                primary = entry.get("concept")
                alts = entry.get("alternatives", []) or []
                candidates = [c for c in ([primary] + alts) if c]

            series_found = pd.Series(dtype=float)

            for tag in candidates:
                try:
                    s = self.extract_series(facts, tag)

                except Exception:
                    s = pd.Series(dtype=float)

                if not s.empty:
                    series_found = s
                    break

            if series_found.empty:
                # if config explicitly requests a forced column, create an empty series aligned to last ~20 quarters
                if isinstance(entry, dict) and entry.get("force_column", False):
                    # create index of last 20 quarters based on other data we have (fallback to empty)
                    # easiest is to build from any existing series in `data`; otherwise leave empty
                    if data:
                        idx = next(iter(data.values())).index
                        data[out_col] = pd.Series(index=idx, dtype=float)

                    else:
                        # empty column; will produce empty DF but column present
                        data[out_col] = pd.Series(dtype=float)

                else:
                    # nothing for this tag -> skip column entirely
                    continue

            else:
                data[out_col] = series_found

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data).sort_index()

        # keep only last ~20 quarters (~5 years)
        return df.iloc[-20:]



    # ─── GETTER METHODS ───────────────────────────────────────────────────────────────────────────────────────────────

    def get_income_statement(self, ticker: str) -> pd.DataFrame:

        """
        Fetches the quarterly income statement for a given symbol.

        :param ticker: The stock ticker symbol, e.g. 'AAPL'.
        :return: A DataFrame of income statement line items.
        """

        debug = self.build_statement(ticker, "income_q")

        return debug



    def get_balance_sheet(self, ticker: str) -> pd.DataFrame:

        """
        Fetches the quarterly balance sheet for a given symbol.

        :param ticker: The stock ticker symbol, e.g. 'AAPL'.
        :return: A DataFrame of balance sheet line items.
        """

        return self.build_statement(ticker, "balance_sheet_q")



    def get_cash_flow(self, ticker: str) -> pd.DataFrame:

        """
        Fetches the quarterly cash flow statement for a given symbol.

        :param ticker: The stock ticker symbol, e.g. 'AAPL'.
        :return: A DataFrame of cash flow line items.
        """

        return self.build_statement(ticker, "cashflow_q")