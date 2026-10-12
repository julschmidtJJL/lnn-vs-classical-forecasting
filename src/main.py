import datetime
import glob
import json
import os
import re
import subprocess
import time
import uuid
import yaml
import pandas as pd
import torch
from itertools import product
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load secrets from the repo-root .env (gitignored). In Docker we pass the same
# variables via `docker run --env-file .env`; there is no .env file in the image
# and load_dotenv is then a harmless no-op (it never overrides vars already set).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a secret/config value from the environment, falling back to default."""
    return os.environ.get(name, default)

# Receive data
from data_import.internal_yahoo import InternalYahooFinanceDataWrapper
from data_import.external import ExternalDataWrapper
from data_import.internal_sec import InternalSECWrapper

# Clean data
from data_cleaning.cleaner import DataCleaner

# Build features & compliance check
from features.feature_builder import FeatureBuilder
from features.stationarize import stationarize_exog_cfg
from data_exploration.compliance import DataExplorationCompliance
from data_exploration.visualisation import DataExplorationVisualisation

# Log results via Supabase
from loggers.supabase_logger import SupabaseLogger
from loggers.resilient_logger import ResilientLogger

# Run models
from models.sarimax_model import SARIMAModel
from models.varmax_model import VARMAXModel
from models.lstm_model import LSTMModel
from models.ltc_model import LTCModel
from models.cfc_model import CfCModel
from models.naive_model import NaiveBaseline
from models.ctrnn_model import CTRNNModel
from models.neural_ode_model import NeuralODEModel
from models.feature_policy import apply_target_policy

# Evaluate results
import numpy as np
from evaluation.evaluator import Evaluator, save_results, dm_test_stat
from validation.walk_forward import WalkForwardSplitter
from validation.data_quality import validate_feature_matrix



# Secrets loaded from .env (see .env.example). Validated at the logger
# construction site so importing this module (e.g. in tests) never fails.
SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_KEY = _env("SUPABASE_KEY")



# ─── RECEIVING DATA ───────────────────────────────────────────────────────────────────────────────────────────────────

def get_internal_data(ticker: str, period: str, interval: str, cfg) -> dict:

    """
    Fetches internal raw data of a specific publicly listed company.

    :param ticker: Ticker symbol of the publicly listed company being analysed, e.g. "AAPL".
    :param period: Period for which data should be fetched, e.g. "5y" or "2mo".
    :param interval: Interval for which data should be fetched, e.g. "1d" or "1mo".
    :param cfg: Your config.yaml file containing all important configurations.
    :return: All the internal company data in one dict.
    """

    raw_internal_yahoo = InternalYahooFinanceDataWrapper(ticker=ticker, period=period, interval=interval)

    sec_user_agent = _env("SEC_USER_AGENT") or cfg["data"]["internal"].get("sec_user_agent")
    raw_internal_sec = InternalSECWrapper(user_agent=sec_user_agent, cfg=cfg)

    # SEC fundamentals exist for individual equities only. For ETFs/indices (e.g. SPY)
    # or anything not in the SEC ticker map, fall back to empty frames and let the
    # feature builder skip those columns (build_financial_features handles empty input).
    def _try_sec(fn_name):
        try:
            return getattr(raw_internal_sec, fn_name)(ticker=ticker)
        except Exception as exc:
            print(f"[get_internal_data] SEC {fn_name} unavailable for {ticker} ({exc}); "
                  "skipping fundamentals for this ticker.")
            return pd.DataFrame()

    return {
        "prices": raw_internal_yahoo.get_price_data(),
        "info": raw_internal_yahoo.get_info(),
        "income_q": _try_sec("get_income_statement"),
        "balance_sheet_q": _try_sec("get_balance_sheet"),
        "cashflow_q": _try_sec("get_cash_flow"),
    }



def get_external_data(period: str, interval: str, cfg) -> dict:

    """
    Fetches all external data in one go:
      - FRED series as DataFrame.
      - Indices history as dict of DataFrames.

    :param period: Period for which data should be fetched, e.g. "5y" or "2mo".
    :param interval: Interval for which data should be fetched, e.g. "1d" or "1mo".
    :param cfg: Config.yaml contents.
    :return: All the external data in one dict.
    """

    ext_cfg = cfg["data"]["external"]

    # 1) FRED macro series
    fred_map = {
        item["name"]: item["series_id"]
        for item in ext_cfg["fred"]["series"]
    }

    fred_api_key = _env("FRED_API_KEY") or ext_cfg["fred"].get("api_key")
    if not fred_api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Copy .env.example to .env and fill it in "
            "(or pass --env-file .env to docker run)."
        )
    edw = ExternalDataWrapper(fred_api_key=fred_api_key, tickers=ext_cfg.get("indices", []))

    raw_fred = edw.get_fred_multiple(fred_map, start_date=ext_cfg["fred"]["start"], end_date=ext_cfg["fred"]["end"])

    # 2) Yahoo indices
    raw_idcs = edw.get_multiple_indices(tickers=ext_cfg.get("indices", []), period=period, interval=interval)

    return {
        "fred": raw_fred,
        "indices": raw_idcs
    }



# ─── CLEANING DATA ────────────────────────────────────────────────────────────────────────────────────────────────────

def get_clean_data(raw_internal, raw_external, cfg) -> tuple:

    """
    Cleans all internal and external datasets.

    :param raw_internal: The raw internal dataset.
    :param raw_external: The raw external dataset.
    :param cfg: Your config.yaml file.
    :return: Both cleaned internal and external datasets.
    """

    cleaner = DataCleaner(cfg)

    # Debug dump — DISABLED 2026-06-08: hardcoded macOS Desktop path crashes on a
    # Linux cloud VM. Commented (not deleted); uncomment for local xlsx inspection.
    # cleaner.export_to_excel(internal=raw_internal, external=raw_external,
    #                         path="/Users/juliusschmidt/Desktop/BachelorThesisRaw.xlsx")

    clean_internal = cleaner.clean_internal(raw_internal)
    clean_external = cleaner.clean_external(raw_external)

    # Debug dump — DISABLED 2026-06-08 (same reason as above).
    # cleaner.export_to_excel(internal=clean_internal, external=clean_external,
    #                         path="/Users/juliusschmidt/Desktop/BachelorThesisClean.xlsx")

    return clean_internal, clean_external



# ─── CREATING FEATURE SET ─────────────────────────────────────────────────────────────────────────────────────────────

def get_features(clean_internal, clean_external, cfg) -> pd.DataFrame:

    """
    Merges both internal and external datasets. Also creates some first features and does a first cleaning.

    :param clean_internal: The cleaned internal dataset.
    :param clean_external: The cleaned external dataset.
    :param cfg: Your config.yaml file.
    :return: The merged datasets, after a first cleaning.
    """

    feature_builder = FeatureBuilder(cfg=cfg)

    df = feature_builder.build_features(clean_internal=clean_internal, clean_external=clean_external)

    df.to_excel("/Users/juliusschmidt/Desktop/BachelorThesisFeatures.xlsx", index="True")

    return df



# ─── CONFIG VALIDATION ────────────────────────────────────────────────────────────────────────────────────────────────

_VALID_MODELS = {"naive", "sarimax", "sarimax_aic", "varmax", "lstm", "ltc", "cfc", "ctrnn", "neural_ode"}
_VALID_CONDITIONS = {"transformed", "raw"}
_CLASSICAL_MODELS = {"naive", "sarimax", "sarimax_aic", "varmax"}  # must not run on raw condition


def validate_experiment_config(cfg: dict) -> None:
    """
    Validate experiment.runs before any data is loaded.
    Raises ValueError with a clear message on the first invalid entry.
    """
    runs = cfg.get("experiment", {}).get("runs", [])

    if not runs:
        raise ValueError("experiment.runs is empty — nothing to evaluate.")

    seen = set()
    for i, run in enumerate(runs):
        model = run.get("model")
        condition = run.get("condition")

        if model not in _VALID_MODELS:
            raise ValueError(
                f"experiment.runs[{i}]: unknown model '{model}'. Valid: {sorted(_VALID_MODELS)}"
            )
        if condition not in _VALID_CONDITIONS:
            raise ValueError(
                f"experiment.runs[{i}]: unknown condition '{condition}'. Valid: {sorted(_VALID_CONDITIONS)}"
            )
        if model in _CLASSICAL_MODELS and condition == "raw":
            raise ValueError(
                f"experiment.runs[{i}]: classical model '{model}' cannot run on condition 'raw'. "
                "Stationarity transforms are required by model assumptions."
            )
        # Variant-aware dedup (run #2): the same base model can appear multiple
        # times under different variant ids (e.g. cfc default_ncp vs cfc pure_fc),
        # so the identity key includes the variant. Non-variant runs keep
        # variant=None, so #0/#1 dedup behaviour is unchanged.
        variant = run.get("variant")
        # A present-but-empty variant ("" / falsy) would pass dedup as a distinct
        # key yet be falsy in run_key generation (f"{model}_{variant}_..." if variant
        # else f"{model}_..."), silently colliding with the bare run_key. Reject it.
        if "variant" in run and not variant:
            raise ValueError(
                f"experiment.runs[{i}]: empty variant for model '{model}'. Omit the "
                "'variant' key for the base model, or give a non-empty variant id."
            )
        key = (model, variant, condition)
        if key in seen:
            raise ValueError(
                f"experiment.runs[{i}]: duplicate entry "
                f"(model='{model}', variant={variant!r}, condition='{condition}')."
            )
        seen.add(key)


_VALID_RUN_KINDS = {"run0", "run1", "run2", "run2b", "smoke"}


def assert_stationarize_input_source(cfg: dict, input_group_id) -> None:
    """Run #2b safety (review P0): ``exog_stationarize`` transforms ``cfg`` at LOAD
    time, but that only affects the LIVE build path (``build_df_final``). When
    ``INPUT_GROUP_ID`` is set, every seed loads a PRE-BUILT input snapshot instead,
    so the stationarizing transform is silently bypassed — the run would train on
    whatever the pinned group holds (e.g. run #0's raw levels) while the stored
    ``config_snapshot`` claims ``exog_stationarize=true``. The byte-hash verify
    can't catch this (the snapshot is internally consistent). So refuse the combo:
    the stationarized arm MUST build its inputs live (``INPUT_GROUP_ID`` unset).
    Escape hatch for the legitimate case of pinning to an EARLIER stationarized
    group: ``ALLOW_PINNED_STATIONARIZED=1``.
    """
    if cfg.get("experiment", {}).get("exog_stationarize", False) and input_group_id \
            and not _env("ALLOW_PINNED_STATIONARIZED"):
        raise RuntimeError(
            f"exog_stationarize=true but INPUT_GROUP_ID={input_group_id!r} is set. The "
            "stationarizing transform only applies when inputs are built live; a pinned "
            "snapshot would be loaded as-is, silently training on non-stationarized "
            "inputs while labelling results as stationarized. Unset INPUT_GROUP_ID so "
            "the run builds (and snapshots) its own stationarized inputs. If the pinned "
            "group was itself built under exog_stationarize, set "
            "ALLOW_PINNED_STATIONARIZED=1 to override."
        )


def assert_run_kind(cfg: dict, config_path: str) -> None:
    """Fail-fast guard against launching the wrong run-list.

    Each config self-declares ``experiment.run_kind``. When the launcher sets the
    ``RUN_KIND`` env var, it MUST equal the loaded config's declared kind, else we
    abort before any Supabase writes. This stops e.g. a bare ``python main.py``
    (which defaults to config.yaml = run #2's variant run-list) or the Docker smoke
    CMD from being launched under run #1's RUN_ID and silently mislabelling results.

    Always prints the loaded kind. Enforcement:
    - RUN_KIND set → must equal the config's declared kind (strict match).
    - RUN_KIND unset but RUN_ID set → ABORT. A named/resumable run (RUN_ID pins the
      result group) must declare its kind, else ``RUN_ID=run1 python main.py`` would
      silently load the default config.yaml (run #2) under run #1's group. Requiring
      RUN_KIND whenever RUN_ID is present is what actually closes that hole.
    - both unset → print only (local/smoke throwaway runs aren't forced to set it).
    """
    declared = (cfg.get("experiment") or {}).get("run_kind")
    expected = _env("RUN_KIND")
    run_id_set = bool(_env("RUN_ID"))
    if declared is not None and declared not in _VALID_RUN_KINDS:
        raise ValueError(
            f"{config_path}: experiment.run_kind={declared!r} is not one of "
            f"{sorted(_VALID_RUN_KINDS)}."
        )
    print(f"[run] config={config_path} run_kind={declared or 'UNDECLARED'} "
          f"RUN_KIND={expected or 'unset'} RUN_ID={'set' if run_id_set else 'unset'}")
    if not expected:
        if run_id_set:
            raise RuntimeError(
                f"RUN_ID is set (a named/resumable run) but RUN_KIND is not. A named "
                f"run must declare its kind so the launcher can't load the wrong "
                f"run-list under this RUN_ID. Set RUN_KIND to match the config you "
                f"intend, e.g. RUN_KIND={declared or '<config run_kind>'} RUN_ID=... "
                f"python main.py {config_path}."
            )
        return
    if declared is None:
        raise RuntimeError(
            f"RUN_KIND={expected!r} is set but {config_path} declares no "
            f"experiment.run_kind. Add 'run_kind: {expected}' to the config's "
            f"experiment block, or unset RUN_KIND."
        )
    if expected != declared:
        raise RuntimeError(
            f"RUN_KIND mismatch: env RUN_KIND={expected!r} but {config_path} "
            f"declares experiment.run_kind={declared!r}. You are about to launch "
            f"the WRONG run-list. Fix the config argument or the RUN_KIND env "
            f"(e.g. run #1 = 'python main.py config.run1.yaml' with RUN_KIND=run1) "
            f"and relaunch."
        )


def build_df_final(clean_internal: dict, clean_external: dict, condition: str, cfg: dict,
                   ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Build the final feature DataFrame for a given preprocessing condition.
    Called once per unique condition; results are cached by the caller.

    ``ticker`` is forwarded to FeatureBuilder so per-ticker exclusions in
    ``cfg["data"]["feature_exclusions"]`` are applied (e.g. drop ^GSPC when
    forecasting SPY to avoid near-direct leakage).
    """
    pp = cfg.get("preprocessing", {})
    macro_lag = int(pp.get("macro_lag", 1))
    warmup = int(pp.get("warmup_discard", 20))

    fb = FeatureBuilder(cfg=cfg)
    df_raw = fb.build_features(
        clean_internal=clean_internal,
        clean_external=clean_external,
        condition=condition,
        macro_lag=macro_lag,
        ticker=ticker,
    )

    # Run #2b: when the exog_stationarize table has already stationarized the
    # macro/index columns (applied in main() before this), create_stationarity must
    # NOT stationarize them again — double log-diffing the indices yields -inf (the
    # smoke-gate launch-blocker). Internal price/financial transforms still run.
    exog_stat = bool((cfg.get("experiment") or {}).get("exog_stationarize", False))
    compliance = DataExplorationCompliance(df_raw)
    return compliance.create_stationarity(condition=condition, warmup_discard=warmup,
                                          skip_exog_stationarization=exog_stat)


# ─── STATIONARITY CHECKS AND FIXES ────────────────────────────────────────────────────────────────────────────────────

def get_compliance_check(df) -> pd.DataFrame:

    """
    Does a visual and compliance (ADFuller, KPSS, ACF, PACF) check. Also creates the final feature set.

    :param df: The pre final feature set.
    :return: The final feature set.
    """

    compliance = DataExplorationCompliance(df)

    # stationarity_results = compliance.stationarity_report()
    # stationarity_results.to_excel("/Users/juliusschmidt/Desktop/BachelorThesisStationarityResults.xlsx")

    # acf_pacf_results = compliance.acf_pacf_report(columns=None)
    # acf_pacf_results.to_excel("/Users/juliusschmidt/Desktop/BachelorThesisAcfPacfResults.xlsx")

    df_final = compliance.create_stationarity()

    df_final.to_excel("/Users/juliusschmidt/Desktop/BachelorThesisFinalFeatures.xlsx")

    visualisation = DataExplorationVisualisation(df_final)

    # visualisation.acf_vis()
    # visualisation.pacf_vis()

    return df_final



# ─── RUNNING ML MODELS ────────────────────────────────────────────────────────────────────────────────────────────────

def build_sarimax_grid(cfg):

    """
    Builds the hyperparameter grid for statsmodels' (S)ARIMA(X) model.

    :param cfg: Your config.yaml file.
    :return: All the order and seasonal order combinations (max. 20).
    """

    r = cfg["models"]["SARIMAX"].get("ranges", {})

    p_vals = r.get("p", [0, 1])
    d_vals = r.get("d", [0, 1])
    q_vals = r.get("q", [0, 1])

    P_vals = r.get("P", [0, 1])
    D_vals = r.get("D", [0, 1])
    Q_vals = r.get("Q", [0, 1])
    s_vals = r.get("s", [int(cfg["models"]["SARIMAX"]["seasonal_order"][3])])

    max_combos = int(r.get("max_combos", 20))

    orders = [(p, d, q) for p in p_vals for d in d_vals for q in q_vals]
    seasonals = [(P, D, Q, s) for P in P_vals for D in D_vals for Q in Q_vals for s in s_vals]

    combos = list(product(orders, seasonals))
    combos = combos[:max_combos]

    return [(tuple(o), tuple(s)) for o, s in combos]



def get_sarimax(df_final, logger, ticker, cfg):

    """
    Runs statsmodels' (S)ARIMA(X) model.

    :param df_final: The final feature set.
    :param logger: The Supabase logger to log all experiment results properly.
    :param ticker: The stock being analysed -> Supabase db.
    :param cfg: Your config.yaml file.
    """

    sarima = SARIMAModel(cfg=cfg, features=df_final, logger=logger, ticker=ticker)

    sarima_y_train, sarima_y_test = sarima.train_test_split()

    combos = build_sarimax_grid(cfg=cfg)

    for i, (order, seasonal_order) in enumerate(combos, start=1):

        sarima_model_fit = sarima.fit(y_train=sarima_y_train, o=order, so=seasonal_order)

        sarima_predictions, sarima_residuals, sarima_exp_id = sarima.predict(model_fit=sarima_model_fit,
                                                                         y_test=sarima_y_test, y_train=sarima_y_train,
                                                                         include_intervals=True)

        sarima.historic_forecast(experiment_id=sarima_exp_id, include_intervals=True, y_test=sarima_y_test)

        sarima.future_forecast(experiment_id=sarima_exp_id, steps=cfg["models"]["SARIMAX"]["forecast_steps"],
                           include_intervals=True, o=order)



def build_varmax_grid(cfg):

    """
    Builds the hyperparameter grid for statsmodels' VARMA(X) model.

    :param cfg: Your config.yaml file.
    :return: All the order and seasonal order combinations (max. 20).
    """

    r = cfg["models"]["VARMAX"].get("ranges", {})

    p_vals = r.get("p", [0, 1])
    q_vals = r.get("q", [0, 1])

    max_combos = int(r.get("max_combos", 20))

    orders = [(p, q) for p in p_vals for q in q_vals]

    combos = orders[:max_combos]

    return [tuple(o) for o in combos]  # returns list of (p,q)



def get_varmax(df_final, logger, ticker, cfg):

    """
    Runs statsmodels' VARMA(X) model.

    :param df_final: The final feature set.
    :param logger: The Supabase logger to log all experiment results properly.
    :param ticker: The stock being analysed -> Supabase db.
    :param cfg: Your config.yaml file.
    """

    varmax = VARMAXModel(cfg=cfg, features=df_final, logger=logger, ticker=ticker)

    varmax_y_train, varmax_y_test = varmax.train_test_split()

    combos = build_varmax_grid(cfg=cfg)

    for i, order in enumerate(combos, start=1):
        try:
            varmax_model_fit = varmax.fit(y_train=varmax_y_train, o=order)

            varmax_predictions, varmax_residuals, varmax_exp_id = varmax.predict(model_fit=varmax_model_fit,
                                                                             y_test=varmax_y_test, y_train=varmax_y_train,
                                                                             include_intervals=True)

            varmax.historic_forecast(experiment_id=varmax_exp_id, include_intervals=True, y_test=varmax_y_test)

            varmax.future_forecast(experiment_id=varmax_exp_id, steps=cfg["models"]["VARMAX"]["forecast_steps"],
                                    include_intervals=True, o=order)

        except Exception as e:
            print(f"VARMAX combo failed order={order}: {e}")
            continue



def build_lstm_grid(cfg):

    """
    Returns list[dict] of LSTM hyperparameter combos (bounded by max_combos).
    Config path: cfg["models"]["LSTM"]["ranges"]
    Example ranges keys: hidden_size, num_layers, dropout, lr, batch_size, seq_len, epochs

    :param cfg: Your config.yaml file.
    :return: All the hyperparameter combinations.
    """

    r = cfg["models"]["LSTM"].get("ranges", {})

    hidden_sizes = r.get("hidden_size", [64])
    num_layers = r.get("num_layers", [1, 2])
    dropouts = r.get("dropout", [0.0])
    lrs = r.get("lr", [0.001])
    batch_sizes = r.get("batch_size", [32])
    seq_lens = r.get("seq_len", [20])
    epochs_list = r.get("epochs", [10])

    max_combos = int(r.get("max_combos", 20))

    combos = list(product(hidden_sizes, num_layers, dropouts, lrs, batch_sizes, seq_lens, epochs_list))
    combos = combos[:max_combos]

    dicts = []

    for hs, nl, dp, lr, bs, sl, ep in combos:
        dicts.append({
            "hidden_size": int(hs),
            "num_layers": int(nl),
            "dropout": float(dp),
            "lr": float(lr),
            "batch_size": int(bs),
            "seq_len": int(sl),
            "epochs": int(ep)
        })

    return dicts



def get_lstm(df_final, logger, ticker, cfg):

    """
    Runs PyTorchs LSTM model.

    :param df_final: The final feature set.
    :param logger: The Supabase logger to log all experiment results properly.
    :param ticker: The stock being analysed -> Supabase db.
    :param cfg: Your config.yaml file.
    """

    lstm = LSTMModel(cfg=cfg, features=df_final, logger=logger, ticker=ticker)

    X_train, X_test, y_train, y_test = lstm.train_test_split()

    lstm_grid = build_lstm_grid(cfg=cfg)

    for i, params in enumerate(lstm_grid, start=1):
        try:
            model = lstm.fit(X_train=X_train, y_train=y_train, params=params, verbose=False)

            # pass X_train to predict
            preds, resid, exp_id = lstm.predict(model_fit=model, X_train=X_train, X_test=X_test, y_train=y_train,
                                                y_test=y_test)

            # DOESN'T MAKE SENSE HERE LIKE WITH STATSMODELS ML MODELS (VARMAX and SARIMAX)
            # lstm.historic_forecast(exp_id, include_intervals=False, X_test=X_test, y_test=y_test, X_train=X_train,
                                   # y_train=y_train)

            lstm.future_forecast(experiment_id=exp_id, steps=cfg["models"]["LSTM"].get("forecast_steps", 10),
                                 params=params)

        except Exception as e:
            print(f"LSTM combo failed: {e}")
            # traceback.print_exc()
            continue



def build_ltc_grid(cfg):

    """
    Build hyperparameter grid for the LTC model. Keys: units, lr, batch_size, seq_len, epochs, sparsity_level, clip_gradient

    :param cfg: Your config.yaml file.
    :return: list[dict] of param dicts (bounded by max_combos).
    """

    r = cfg["models"]["LTC"].get("ranges", {})

    units_list = r.get("units", [16])
    lrs = r.get("lr", [0.01])
    batch_sizes = r.get("batch_size", [32])
    seq_lens = r.get("seq_len", [20])
    epochs_list = r.get("epochs", [10])
    sparsities = r.get("sparsity_level", [0.5])
    clip_grads = r.get("clip_gradient", [1])

    max_combos = int(r.get("max_combos", 20))

    combos = list(product(units_list, lrs, batch_sizes, seq_lens, epochs_list, sparsities, clip_grads))
    combos = combos[:max_combos]

    dicts = []
    for units, lr, bs, sl, ep, sp, cg in combos:
        dicts.append({
            "units": int(units),
            "lr": float(lr),
            "batch_size": int(bs),
            "seq_len": int(sl),
            "epochs": int(ep),
            "sparsity_level": float(sp),
            "clip_gradient": float(cg)
        })

    return dicts



def get_ltc(df_final, logger, ticker, cfg):

    """
    Runs NCPs LTC model.

    :param df_final: The final feature set.
    :param logger: The Supabase logger to log all experiment results properly.
    :param ticker: The stock being analysed -> Supabase db.
    :param cfg: Your config.yaml file.
    """

    ltc = LTCModel(cfg=cfg, features=df_final, logger=logger, ticker=ticker)

    X_train, X_test, y_train, y_test = ltc.train_test_split()

    ltc_grid = build_ltc_grid(cfg=cfg)

    for i, params in enumerate(ltc_grid, start=1):
        try:
            model = ltc.fit(X_train=X_train, y_train=y_train, params=params, verbose=False)

            preds, resid, exp_id = ltc.predict(model_fit=model, X_train=X_train, X_test=X_test, y_train=y_train,
                                               y_test=y_test)

            ltc.future_forecast(experiment_id=exp_id, steps=cfg["models"]["LTC"].get("forecast_steps", 10),
                                params=params)

        except Exception as e:
            print(f"LTC combo failed: {e}")
            # traceback.print_exc()
            continue



def build_cfc_grid(cfg):

    """
    Build hyperparameter grid for the CfC model.

    :param cfg: Your config.yaml file.
    :return: list[dict] of param dicts (bounded by max_combos).
    """

    r = cfg["models"]["CfC"].get("ranges", {})

    units_list = r.get("units", [16])
    lrs = r.get("lr", [0.01])
    batch_sizes = r.get("batch_size", [32])
    seq_lens = r.get("seq_len", [20])
    epochs_list = r.get("epochs", [10])
    sparsities = r.get("sparsity_level", [0.5])
    clip_grads = r.get("clip_gradient", [1])

    max_combos = int(r.get("max_combos", 20))

    combos = list(product(units_list, lrs, batch_sizes, seq_lens, epochs_list, sparsities, clip_grads))
    combos = combos[:max_combos]

    dicts = []
    for units, lr, bs, sl, ep, sp, cg in combos:
        dicts.append({
            "units": int(units),
            "lr": float(lr),
            "batch_size": int(bs),
            "seq_len": int(sl),
            "epochs": int(ep),
            "sparsity_level": float(sp),
            "clip_gradient": float(cg)
        })

    return dicts



def get_cfc(df_final, logger, ticker, cfg):

    """
    Runs NCPs CfC model.

    :param df_final: The final feature set.
    :param logger: The Supabase logger to log all experiment results properly.
    :param ticker: The stock being analysed -> Supabase db.
    :param cfg: Your config.yaml file.
    """

    cfc = CfCModel(cfg=cfg, features=df_final, logger=logger, ticker=ticker)

    X_train, X_test, y_train, y_test = cfc.train_test_split()

    cfc_grid = build_cfc_grid(cfg=cfg)

    for i, params in enumerate(cfc_grid, start=1):
        try:
            model = cfc.fit(X_train=X_train, y_train=y_train, params=params, verbose=False)

            preds, resid, exp_id = cfc.predict(model_fit=model, X_train=X_train, X_test=X_test, y_train=y_train,
                                               y_test=y_test)

            cfc.future_forecast(experiment_id=exp_id, steps=cfg["models"]["CfC"].get("forecast_steps", 10),
                                params=params)

        except Exception as e:
            print(f"LTC combo failed: {e}")
            # traceback.print_exc()
            continue


# ─── WALK-FORWARD EVALUATION ──────────────────────────────────────────────────────────────────────────────────────────

_FOLD_SUCCESS_THRESHOLD = 0.90


def _collect_model_folds(model_key, folds, y_series, X_df, fit_fn, predict_fn,
                         success_threshold: float = _FOLD_SUCCESS_THRESHOLD,
                         meta_fn=None):
    """
    Run walk-forward folds for one model and return per-fold results.

    fit_fn(train_idx)                         → model_fit
    predict_fn(fit, train_idx, test_idx)      → (predictions_array, residuals_array)
    meta_fn(fit)  (optional)                  → dict of per-fold metadata
        ({"params", "runtime_seconds", "diagnostics"}) carried into
        experiment_results by _log_folds_centrally. Restores the per-fold
        params / runtime / classical AIC-BIC that the per-model log_result
        used to write before logging was centralized. Best-effort: a failure
        here never fails the fold.

    Returns dict {fold_id: {"y_true", "y_pred", "residuals" [, "meta"]}} for
    successfully completed folds. Per the codex review, if the success rate
    falls below ``success_threshold`` (default 90%), the model is EXCLUDED
    entirely (empty dict returned) so it does not enter the DM tables with
    sparse, under-powered data. A loud warning is printed for the run summary.
    """
    results = {}
    total = len(folds)
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        try:
            fit = fit_fn(train_idx)
            preds, resids = predict_fn(fit, train_idx, test_idx)
            preds = np.asarray(preds, dtype=float)
            resids = np.asarray(resids, dtype=float)
            # A non-converged fit (notably fixed-order MA/ARMA with
            # enforce_invertibility=False) can return NaN/inf forecasts WITHOUT
            # raising. Admitting them as a "successful" fold poisons the DM tables
            # with NaN and, for primary models, shrinks the aligned fold set for
            # every other primary model. Treat a non-finite fold as a failed fold
            # so it drops here and counts against the success_threshold.
            if not (np.isfinite(preds).all() and np.isfinite(resids).all()):
                raise ValueError(
                    f"non-finite forecast/residuals ({int((~np.isfinite(preds)).sum())} "
                    f"NaN/inf preds) — failing fold rather than admitting NaN")
            entry = {
                "y_true": y_series.iloc[test_idx].values,
                "y_pred": preds,
                "residuals": resids,
            }
            if meta_fn is not None:
                try:
                    entry["meta"] = meta_fn(fit)
                except Exception as mexc:
                    print(f"[WF] {model_key} fold {fold_id} meta capture failed: {mexc}")
            results[fold_id] = entry
        except Exception as exc:
            print(f"[WF] {model_key} fold {fold_id} failed: {exc}")

    if total > 0 and (len(results) / total) < success_threshold:
        print(
            f"[WF] EXCLUDING {model_key}: only {len(results)}/{total} folds succeeded "
            f"(< {int(success_threshold * 100)}% threshold). Model will not enter DM tables."
        )
        return {}

    return results


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs (and Lightning's, if present) so a
    given seed reproduces the same neural fits. Best-effort: missing optional
    deps are ignored."""
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import pytorch_lightning as _pl
        _pl.seed_everything(seed, workers=True)
    except Exception:
        pass


class _FoldSink:
    """No-op logger handed to models during walk-forward. Per-fold logging is
    owned centrally by ``_log_folds_centrally``; the in-model log_result path
    was deleted, so this is a defensive guard: if a log_result call ever
    reappears in a model's predict(), it lands here instead of double-writing
    to experiment_results. summary/forecast methods are absent on purpose —
    walk-forward only ever exercised log_result.
    """

    def log_result(self, *args, **kwargs):
        return None


# Display labels for the experiment_results.model_type column, matching the
# labels the models used to log themselves (before logging was centralized).
_MODEL_TYPE_LABELS = {
    "naive": "Naive", "sarimax": "SARIMAX", "varmax": "VARMAX", "lstm": "LSTM",
    "ltc": "LTC", "cfc": "CfC", "ctrnn": "CTRNN", "neural_ode": "NeuralODE",
}


def _fold_metrics(y_true, y_pred, y_train=None) -> dict:
    """Per-fold point metrics, replicating the formulas the models used to
    compute inline before logging was centralized (eps=1e-9, finite-mask,
    diff-MASE on the training series). Kept identical so the centralized
    experiment_results rows match the historical per-model values.
    """
    eps = 1e-9
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    finite = np.isfinite(yp) & np.isfinite(yt)

    if finite.sum() == 0:
        return {"mse": None, "mae": None, "rmse": None, "smape": None,
                "mase": None, "directional_accuracy": None}

    yp, yt = yp[finite], yt[finite]
    mse = float(np.mean((yt - yp) ** 2))
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(mse))
    smape = float(100.0 * np.mean(2.0 * np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + eps)))
    da = float(np.mean(np.sign(yp) == np.sign(yt)))

    mase = None
    if y_train is not None:
        tr = np.asarray(y_train, dtype=float).reshape(-1)
        if tr.size >= 2:
            scale_errors = np.abs(tr[1:] - tr[:-1])
            scale = float(np.mean(scale_errors)) if scale_errors.size > 0 else eps
            mase = float(mae / (scale + eps))

    return {"mse": mse, "mae": mae, "rmse": rmse, "smape": smape,
            "mase": mase, "directional_accuracy": da}


def _log_folds_centrally(logger, all_results: dict, folds: list, df_finals: dict,
                         ticker: str, run_meta: dict, seed: int, run_identity=None) -> None:
    """Single source of per-fold logging (the per-model log_result calls and
    fold counters were deleted). Uses the REAL walk-forward fold index that
    keys ``all_results`` and batch-upserts one call per (model, condition) run.

    run_id/condition/seed/fold_id are stamped here, so re-running the same
    RUN_ID upserts rather than duplicates (migration 002 unique index).
    """
    run_id = run_meta.get("run_id")
    for run_key, fold_results in all_results.items():
        if not fold_results:
            continue
        ident = (run_identity or {}).get(run_key)
        if ident:
            base_model, variant, condition = ident["base_model"], ident.get("variant"), ident["condition"]
        else:
            base_model, condition = run_key.rsplit("_", 1)
            variant = None
        label = _MODEL_TYPE_LABELS.get(base_model, base_model)
        # model_type is part of the experiment_results upsert conflict key
        # (run_id,ticker,model_type,condition,seed,fold_id — see SupabaseLogger
        # _RESULTS_CONFLICT_COLS). Run #2 variants share a base label (all CfC →
        # "CfC"), so without a distinct model_type the 6 CfC / 2 LTC variants
        # collide on that key and overwrite each other's per-fold rows. Use the
        # unique run_key for variant runs; non-variant #0/#1 keep the base label.
        model_type_value = run_key if variant else label
        y_cond = df_finals[condition].reset_index(drop=True)["log_return"].astype(float)

        rows = []
        for fold_id, fr in fold_results.items():
            train_idx, test_idx = folds[fold_id]
            m = _fold_metrics(fr["y_true"], fr["y_pred"], y_cond.iloc[train_idx])
            # Per-fold metadata (params / runtime / classical AIC-BIC) captured
            # by the model's meta_fn, merged back into the same columns the
            # legacy per-model log_result wrote. Absent for models with no
            # meta_fn (e.g. naive) — those keep the minimal params/diagnostics.
            meta = fr.get("meta") or {}
            params = {"model": label, "variant": variant, "run_key": run_key,
                      "condition": condition, **(meta.get("params") or {})}
            diagnostics = {"mse": m["mse"], "mae": m["mae"], "rmse": m["rmse"],
                           **(meta.get("diagnostics") or {})}
            rows.append({
                "model_type": model_type_value,
                "params": params,
                "train_start": str(int(train_idx[0])),
                "train_end": str(int(train_idx[-1])),
                "test_start": str(int(test_idx[0])),
                "test_end": str(int(test_idx[-1])),
                "fold_id": int(fold_id),
                "horizon": int(len(test_idx)),
                "ticker": ticker,
                "run_id": run_id,
                "condition": condition,
                "seed": int(seed),
                "runtime_seconds": meta.get("runtime_seconds"),
                "notes": f"{label} run (central log)",
                "diagnostics": diagnostics,
                **m,
            })
        try:
            logger.log_results_batch(rows)
        except Exception as exc:  # ResilientLogger handles fallback; guard direct SupabaseLogger use
            print(f"[WF] central fold logging failed for {run_key}: {exc}")


_SNAPSHOT_BUCKET = "experiment-snapshots"


def _input_parquet_path(group, ticker, cond):
    return f"{group}/{ticker}/input_{cond}.parquet"


def _input_fp_path(group, ticker, cond):
    return f"{group}/{ticker}/input_{cond}.fingerprint.json"


def _safe_object_token(token):
    """Sanitize a single Supabase Storage object-key token.

    Supabase Storage silently TRUNCATES object keys at characters it treats as
    URL-special — notably ``#`` (fragment) — and the upload still returns HTTP
    200, so the failure is invisible. The per-seed run_id embeds ``#s{seed}``
    (e.g. ``grp#s42``), which previously collapsed all seeds onto one key
    (last-writer-wins) and clobbered seeds 42-45. Map anything outside a safe
    allowlist to ``_`` so every seed gets a distinct, valid key.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(token))


def _predictions_path(group, ticker, run_id):
    # Sanitize EVERY path segment, not just run_id: any '#' (or other URL-special
    # char) anywhere in the key truncates the whole object name in Supabase
    # Storage, so a raw group/ticker would reintroduce the same silent-collision
    # bug. group/ticker are controlled today (no '#'), but defend the invariant.
    g, t, r = _safe_object_token(group), _safe_object_token(ticker), _safe_object_token(run_id)
    return f"{g}/{t}/predictions_{r}.parquet"


# Run #2 T4 safeguard: every forecast snapshot is also written here, locally,
# alongside the Storage upload. If Storage silently drops an object (the run #1
# loss) OR the box dies before reconciliation, the forecasts still exist on disk.
_FORECAST_MIRROR_DIR = os.path.join("results", "forecasts")


def _mirror_predictions_local(data: bytes, group, ticker, run_id) -> str:
    """Write the forecast parquet bytes to ``results/forecasts/<key>`` locally,
    mirroring the Storage key. Independent of the Storage upload (runs even when
    the upload succeeds) so there is always an offline copy of the predictions."""
    rel = _predictions_path(group, ticker, run_id)  # sanitized group/ticker/predictions_*.parquet
    local_path = os.path.join(_FORECAST_MIRROR_DIR, rel)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    return local_path


def predictions_to_fold_map(df: pd.DataFrame) -> dict:
    """Reconstruct ``{run_key: {fold_id: {y_true, y_pred, residuals}}}`` from a
    predictions snapshot DataFrame — the inverse of the row-flattening in
    ``_persist_predictions``.

    Run #2 T4 light enabler for per-unit resume: a future resume path can reload
    already-computed units from their persisted predictions instead of re-running
    them. NOT wired into the DM flow yet (the in-memory family build is unchanged);
    this just makes the reload available + tested so the wiring is a small step.
    """
    out: dict = {}
    for run_key, g in df.groupby("run_key"):
        folds: dict = {}
        for fold_id, fg in g.groupby("fold_id"):
            fg = fg.sort_values("t")
            # Keep residuals aligned 1:1 with t (missing -> NaN), so the reload is a
            # faithful inverse of the stored frame: no length skew, no shifted /
            # dropped interior values (review Medium).
            folds[int(fold_id)] = {
                "y_true": fg["y_true"].to_numpy(dtype=float),
                "y_pred": fg["y_pred"].to_numpy(dtype=float),
                "residuals": fg["residual"].to_numpy(dtype=float),
            }
        out[str(run_key)] = folds
    return out


def _filter_runs_by_models(runs: list, run_models_env) -> list:
    """Filter the run-list to the model families named in ``RUN_MODELS`` (comma-
    separated, case-insensitive, matched on the run's ``model``). Returns ``runs``
    unchanged when the env is empty.

    Run #2 T4 light batch-by-family: launch one family at a time (e.g.
    ``RUN_MODELS=cfc``) for the T5 smoke or to re-run a single family. NOTE: a
    filtered run produces a PARTIAL DM family — batches must use DISTINCT RUN_IDs
    and be combined later via ``analysis.merge_dm`` (do NOT reuse one RUN_ID across
    batches; the per-(ticker,seed) summary-resume would skip later batches).
    """
    if not run_models_env:
        return runs
    keep = {m.strip().lower() for m in str(run_models_env).split(",") if m.strip()}
    return [r for r in runs if str(r.get("model", "")).lower() in keep]


def _resume_enabled(scenarios, run_models_env) -> bool:
    """Whether the per-(ticker,seed) summary-resume skip is safe.

    Safe only for a SINGLE scenario (experiment_summaries has no scenario column,
    so multi-scenario keys would collide) AND when NOT running a filtered family
    batch (review High): the summary is keyed (run_id, ticker), so a CFC-only batch
    would write a summary that wrongly marks the ticker complete and skip a later
    LTC batch under the same RUN_ID. Filtered batches must not auto-skip.
    """
    return len(scenarios) == 1 and not run_models_env


def _download_with_retry(logger, path, attempts=3):
    """Download a snapshot object, retrying transient failures. Raises after
    ``attempts`` (eng-review I5: seeds 2-N hard-abort if the group snapshot is
    unreachable — they must not silently re-fetch divergent live data)."""
    last = None
    for i in range(attempts):
        try:
            return logger.download_snapshot(_SNAPSHOT_BUCKET, path)
        except Exception as exc:
            last = exc
            print(f"[WF] snapshot download retry {i + 1}/{attempts} for {path}: {exc}")
    raise RuntimeError(f"snapshot download failed after {attempts} attempts: {path}") from last


def _gate_feature_matrix(df, ticker, cond):
    """Pre-training data-quality gate on a resolved feature matrix.

    Catches the silent-degradation bug class (all-NaN columns, daily series
    downsampled to monthly, broken index) that built cleanly but would corrupt
    training inputs — e.g. the 2026-06-25 FRED freq-inference bug. Runs on BOTH
    paths: a fresh live build AND a loaded pinned snapshot (so launching on a
    known-bad frozen snapshot hard-stops here). `validate_feature_matrix` raises
    DataQualityError on any ERROR unless ALLOW_DATA_QUALITY_FAIL=1 is set (the
    deliberate-override escape, mirroring ALLOW_NO_GIT_COMMIT). Zero-variance /
    high-NaN are non-fatal warnings; we print a one-line count so the grid log
    stays readable (full detail via scripts/data_quality_check.py)."""
    issues = validate_feature_matrix(df, context=f"{ticker}/{cond}")  # raises on error
    warns = [i for i in issues if i.level == "warn"]
    print(f"[dq] {ticker}/{cond}: data-quality gate OK "
          f"({df.shape[1]} cols, {len(warns)} warning(s)).")


def _resolve_df_finals(logger, clean_internal, clean_external, conditions, cfg,
                       ticker, run_meta):
    """Build (live) or load (snapshot) the per-condition feature frames.

    live  : build_df_final, fingerprint, upload parquet + fingerprint sidecar.
    snapshot (seeds 2..N): download the group's parquet + sidecar, verify the
        parquet BYTE hash and the canonical fingerprint, then load. A mismatch
        hard-aborts (eng-review I5 — wrong input data is never tolerated).

    Returns (df_finals, fingerprints, snapshot_uris).
    """
    import json as _json
    from provenance import (FingerprintMismatch, compute_frame_fingerprint,
                            frame_to_parquet_bytes, parquet_bytes_to_frame,
                            sha256_bytes, verify_frame_against_fingerprint)

    data_source = (run_meta or {}).get("data_source", "live")
    # Where this run WRITES its own input snapshot (live seed-1 upload path).
    write_group = (run_meta or {}).get("run_group_id") or (run_meta or {}).get("run_id") or "nogroup"
    # Where input snapshots are READ from. Normally identical to write_group, but
    # run #1/#2 pin inputs to run #0's frozen group (input_group_id) so every run
    # sees byte-identical inputs while writing its results under its own group.
    # Falls back to write_group, so a normal single-run launch is unchanged.
    read_group = (run_meta or {}).get("input_group_id") or write_group
    df_finals, fingerprints, uris = {}, {}, {}

    for cond in sorted(conditions):
        if data_source == "snapshot":
            pq_path, fp_path = _input_parquet_path(read_group, ticker, cond), _input_fp_path(read_group, ticker, cond)
            data = _download_with_retry(logger, pq_path)
            sidecar = _json.loads(_download_with_retry(logger, fp_path).decode("utf-8"))
            if sha256_bytes(data) != sidecar.get("parquet_sha256"):
                raise FingerprintMismatch(f"{ticker}/{cond}: snapshot parquet byte-hash mismatch")
            df = parquet_bytes_to_frame(data)
            verify_frame_against_fingerprint(df, sidecar.get("fingerprint", {}))
            df_finals[cond], fingerprints[cond] = df, sidecar.get("fingerprint")
            uris[cond] = f"{_SNAPSHOT_BUCKET}/{pq_path}"
            print(f"[WF] loaded + verified input snapshot {ticker}/{cond}.")
        else:
            df = build_df_final(clean_internal, clean_external, cond, cfg, ticker=ticker)
            fp = compute_frame_fingerprint(df)
            data = frame_to_parquet_bytes(df)
            sidecar = {"parquet_sha256": sha256_bytes(data), "fingerprint": fp}
            uri = logger.upload_snapshot(_SNAPSHOT_BUCKET, _input_parquet_path(write_group, ticker, cond),
                                         data, "application/octet-stream")
            fp_uri = logger.upload_snapshot(_SNAPSHOT_BUCKET, _input_fp_path(write_group, ticker, cond),
                                            _json.dumps(sidecar, default=str).encode("utf-8"), "application/json")
            # Multi-seed handoff is load-bearing: seeds 2..N DOWNLOAD this exact
            # snapshot (download_snapshot has no fallback, by design). If either
            # upload soft-failed to a local:// path, the remote object the later
            # seeds fetch does not exist, so they would hard-abort — and on a
            # RUN_ID resume seed 1 is skipped (summary already written), so the
            # run can never self-heal. Fail fast HERE, before seed 1 completes,
            # rather than silently shipping a broken handoff. Single-seed runs
            # never download, so a local:// fallback there is harmless.
            if (run_meta or {}).get("multi_seed") and (
                    str(uri).startswith("local://") or str(fp_uri).startswith("local://")):
                raise RuntimeError(
                    f"Multi-seed input snapshot for {ticker}/{cond} could not be uploaded to "
                    f"Storage (got local fallback: {uri!r} / {fp_uri!r}). Seeds 2..N would be "
                    "unable to load it and the run cannot resume past seed 1. Aborting so the "
                    "snapshot is never silently lost. Fix Storage access and re-launch."
                )
            df_finals[cond], fingerprints[cond], uris[cond] = df, fp, uri
    return df_finals, fingerprints, uris


def _persist_predictions(logger, all_results, ticker, run_meta, seed, run_identity=None):
    """Write per-fold predictions (y_true/y_pred/residual) to a parquet snapshot.

    Enables thesis-grade fold diagnostics + any later DM variant without
    re-running. Soft-fail (upload errors fall back to a local file).

    Identity columns: ``model`` is the BASE model (e.g. cfc), ``variant`` is the
    run #2 variant id (None for #0/#1), and ``run_key`` is the canonical unique
    key. Base model + variant come from ``run_identity`` rather than splitting the
    (variant-laden) run_key on '_' (eng-review Codex #5/#6). For non-variant runs
    the ``model``/``condition`` columns are unchanged; ``variant``/``run_key`` are
    additive."""
    from provenance import frame_to_parquet_bytes
    group = (run_meta or {}).get("run_group_id") or (run_meta or {}).get("run_id")
    run_id = (run_meta or {}).get("run_id")
    rows = []
    for run_key, fr_map in all_results.items():
        if not fr_map:
            continue
        ident = (run_identity or {}).get(run_key)
        if ident:
            base_model, variant, cond = ident["base_model"], ident.get("variant"), ident["condition"]
        else:
            base_model, cond = run_key.rsplit("_", 1)
            variant = None
        for fold_id, fr in fr_map.items():
            yt, yp, rs = fr["y_true"], fr["y_pred"], fr["residuals"]
            for t in range(len(yt)):
                rows.append({
                    "run_group_id": group, "run_id": run_id, "seed": int(seed),
                    "ticker": ticker, "model": base_model, "variant": variant,
                    "run_key": run_key, "condition": cond,
                    "fold_id": int(fold_id), "t": int(t),
                    "y_true": float(yt[t]), "y_pred": float(yp[t]),
                    "residual": float(rs[t]) if t < len(rs) else None,
                })
    if not rows:
        return None
    data = frame_to_parquet_bytes(pd.DataFrame(rows))
    # T4 local mirror: always write an offline copy first, so the forecasts
    # survive even if Storage drops the object or the box dies before reconcile.
    try:
        mirror = _mirror_predictions_local(data, group, ticker, run_id)
        print(f"[WF] forecast snapshot mirrored locally → {mirror}")
    except Exception as exc:
        print(f"[WF] local forecast mirror failed for {ticker}: {exc}")
    try:
        # upload_snapshot verifies-after-write (T4); a silent drop raises and the
        # ResilientLogger wrapper saves its own local fallback.
        return logger.upload_snapshot(_SNAPSHOT_BUCKET, _predictions_path(group, ticker, run_id), data)
    except Exception as exc:
        print(f"[WF] predictions snapshot upload failed for {ticker}: {exc}")
        return None


def _resolve_sarimax_order(run: dict, sarimax_cfg: dict):
    """Resolve (order, seasonal_order) for a SARIMAX run-list entry.

    Run #2 adds explicit fixed-order variant rows (e.g. ``ma1`` -> (0,0,1),
    ``arma11`` -> (1,0,1)) that carry ``params.order``. Run-level params win; when
    absent we fall back to the fixed config default (``models.SARIMAX.order``), so
    the bare ``sarimax`` run_key keeps its #0/#1 AR(1) identity unchanged. This
    mirrors the CfC/LTC variant-override pattern (``params.update(run.get("params"))``).

    :param run: a single experiment.runs entry.
    :param sarimax_cfg: cfg["models"]["SARIMAX"].
    :return: (order, seasonal_order) as tuples of ints.
    """
    params = run.get("params") or {}
    # Fallback is the documented AR(1) identity (1,0,0), NOT (1,1,1): if the
    # config key were ever stripped, the bare `sarimax` headline baseline must
    # not silently drift into an integrated ARIMA(1,1,1).
    order = tuple(params.get("order", sarimax_cfg.get("order", [1, 0, 0])))
    seasonal_order = tuple(params.get("seasonal_order",
                                      sarimax_cfg.get("seasonal_order", [0, 0, 0, 0])))
    return order, seasonal_order


def run_walk_forward_evaluation(clean_internal: dict, clean_external: dict,
                                logger, ticker: str, cfg) -> tuple:
    """
    Dual-condition walk-forward evaluation over all experiment.runs.

    Builds one df_final per unique condition (cached), runs every (model, condition)
    pair, aligns primary fold set (drop folds where any C1 model failed), then:
      - Primary DM + Holm-Bonferroni on the 6 Condition-1 models (15 pairs).
      - Secondary DM (uncorrected) on each neural model's C1 vs C2 pair.

    Returns (metrics_dict, primary_sig_table, secondary_sig_table).
    """
    wf_cfg         = cfg.get("walk_forward", {})
    initial_window = int(wf_cfg.get("initial_window", 750))
    horizon        = int(wf_cfg.get("horizon", 21))
    output_dir     = wf_cfg.get("output_dir", "results/walk_forward")
    # Seed is set per sub-run in main() (multi-seed loop) and carried on
    # _run_meta; fall back to experiment.seed for a single-seed run.
    seed           = int((cfg.get("_run_meta") or {}).get(
                         "seed", cfg.get("experiment", {}).get("seed", 42)))
    runs           = cfg["experiment"]["runs"]

    # T4 light batch-by-family: RUN_MODELS=cfc,ltc restricts the run-list to those
    # model families (T5 smoke / re-run one family). A filtered run is a PARTIAL DM
    # family — use a DISTINCT RUN_ID per batch and merge later (analysis.merge_dm).
    run_models_env = _env("RUN_MODELS")
    if run_models_env:
        _before = len(runs)
        runs = _filter_runs_by_models(runs, run_models_env)
        if not runs:
            raise RuntimeError(
                f"RUN_MODELS={run_models_env!r} matched no rows in the run-list "
                f"(models present: {sorted({r['model'] for r in cfg['experiment']['runs']})}).")
        print(f"[WF] RUN_MODELS={run_models_env!r} → running {len(runs)}/{_before} run-list "
              "rows (PARTIAL DM family; use a distinct RUN_ID per batch + merge_dm later).")

    # Determinism: seed every RNG that touches the neural fits so the per-seed
    # results are reproducible (eng-review I6 / C3). Full bitwise determinism is
    # not guaranteed across machines; CPU-only narrows the gap.
    _seed_everything(seed)

    # ── Build (live) or load (snapshot) df_final per unique condition ─────────────
    # Seed 1 of a run-group builds from live data and uploads the parquet
    # snapshot + fingerprint; seeds 2..N load that exact snapshot so every seed
    # is compared on identical inputs (eng-review I3). data_source / run_group_id
    # come from _run_meta (set per seed in main()).
    run_meta = cfg.get("_run_meta", {})
    conditions_needed = {r["condition"] for r in runs}
    print(f"\n[WF] Resolving feature frames ({run_meta.get('data_source', 'live')}) "
          f"for conditions: {sorted(conditions_needed)}")
    df_finals, input_fingerprints, snapshot_uris = _resolve_df_finals(
        logger, clean_internal, clean_external, conditions_needed, cfg, ticker, run_meta)

    # Data-quality gate: validate every resolved feature matrix (live build OR
    # loaded pinned snapshot) before any model sees it. Catches the silent-
    # degradation bug class — all-NaN columns, daily series downsampled to
    # monthly, broken index — e.g. the 2026-06-25 FRED freq-inference bug.
    # Raises DataQualityError unless ALLOW_DATA_QUALITY_FAIL=1; so launching on a
    # known-bad frozen snapshot hard-stops here unless deliberately overridden.
    for cond, df in sorted(df_finals.items()):
        _gate_feature_matrix(df, ticker, cond)

    # All conditions must have the same row count for DM comparisons to be valid
    row_counts = {cond: len(df) for cond, df in df_finals.items()}
    if len(set(row_counts.values())) > 1:
        raise RuntimeError(
            f"Conditions produced different row counts after warmup discard: {row_counts}. "
            "Check preprocessing.warmup_discard — both conditions must trim the same number of rows."
        )

    # Use transformed frame to define fold structure (same for both conditions)
    ref_df    = df_finals["transformed"]
    splitter  = WalkForwardSplitter(initial_window=initial_window, horizon=horizon)
    folds     = splitter.split(len(ref_df))
    n_folds   = len(folds)
    print(f"[WF] {n_folds} folds (initial_window={initial_window}, horizon={horizon})")

    # ── Run all (model, condition) pairs and collect fold-level results ───────────
    scaler_cfg   = cfg.get("preprocessing", {}).get("scaler", {})
    all_results  = {}   # run_key → {fold_id: {y_true, y_pred, residuals}}
    # run_key → {base_model, variant, condition, primary}. Explicit identity so
    # central logging / persistence do not have to re-derive the base model from
    # the (now variant-laden) run_key via a fragile rsplit (eng-review Codex #5/#6).
    run_identity = {}
    # Models log nothing themselves now; per-fold logging is centralized below.
    fold_sink = _FoldSink()

    for run in runs:
        model     = run["model"]
        condition = run["condition"]
        variant   = run.get("variant")
        # Variant-aware run_key (run #2): a base model can appear under several
        # variants, so the key embeds the variant. No variant ⇒ unchanged #0/#1 key.
        run_key   = f"{model}_{variant}_{condition}" if variant else f"{model}_{condition}"
        # Primary family membership. Default preserves #0/#1 exactly: the
        # transformed (Condition-1) runs are primary, the raw (Condition-2) runs
        # are secondary. Run #2 variants are all transformed ⇒ primary by default;
        # a run may override with an explicit `primary:` flag.
        run_identity[run_key] = {"base_model": model, "variant": variant,
                                 "condition": condition,
                                 "primary": bool(run.get("primary", condition == "transformed"))}
        df        = df_finals[condition]
        df_reset  = df.reset_index(drop=True)
        # INVARIANT: X_df and y_series share row index (X_df.iloc[t] is contemporaneous
        # with y_series.iloc[t]). The one-step-ahead offset is imposed downstream by
        # each model, NOT here. Specifically:
        #   - Neural models (lstm/ltc/cfc/ctrnn/neural_ode) build sliding windows as
        #     X[i:i+seq_len] -> y[i+seq_len], so the window ends at row t-1 when
        #     predicting y_t (see lstm_model.py:343-344, cfc_model.py:586-592).
        #   - Classical models (sarimax/varmax) ignore X and do rolling 1-step-ahead
        #     within each test fold via results.forecast(1) + results.append(y_obs,
        #     refit=False) (see sarimax_model.py:predict, varmax_model.py:predict).
        # All models therefore produce daily 1-step-ahead predictions within a fold;
        # the splitter's horizon=21 controls fold size, not forecast horizon.
        # Do not "fix" the apparent contemporaneity here without auditing every model.
        y_series  = df_reset["log_return"].astype(float)
        # Feature policy (run #0 vs #1): drop log_return unless the experiment
        # opts in via include_past_target_as_feature. apply_target_policy is the
        # single source of truth shared with each model.fit so the two gating
        # layers can never disagree. Classical models ignore X_df regardless.
        X_df      = apply_target_policy(df_reset, cfg)
        scaler_kind = scaler_cfg.get(condition, "zscore")

        # Provenance: log the exact feature set per run so #0/#1 are auditable.
        print(f"\n[WF] Running {run_key} ... ({X_df.shape[1]} feature cols, "
              f"log_return_as_feature={'log_return' in X_df.columns})")

        if model == "naive":
            naive = NaiveBaseline(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)
            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti: None,
                predict_fn=lambda fit, ti, si: naive.predict_fold(y_series.iloc[ti], y_series.iloc[si]),
            )

        elif model == "sarimax":
            sarimax_cfg   = cfg["models"]["SARIMAX"]
            # Run #2: fixed-order variant rows (ma1/arma11) carry params.order;
            # bare `sarimax` falls back to the config AR(1) default (identity
            # preserved for #0/#1 comparability). See _resolve_sarimax_order.
            order, seas   = _resolve_sarimax_order(run, sarimax_cfg)
            sarima        = SARIMAModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)

            def _sarima_predict(fit, ti, si, _y=y_series):
                preds, resids, _ = sarima.predict(
                    model_fit=fit, y_test=_y.iloc[si], y_train=_y.iloc[ti])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _y=y_series: sarima.fit(
                    y_train=_y.iloc[ti], o=order, so=seas),
                predict_fn=_sarima_predict,
                meta_fn=lambda fit: {
                    "params": {"order": list(order),
                               "seasonal_order": list(seas)},
                    "runtime_seconds": float(getattr(sarima, "runtime_seconds", 0) or 0),
                    "diagnostics": {
                        "aic": float(getattr(fit, "aic", float("nan"))),
                        "bic": float(getattr(fit, "bic", float("nan"))),
                        "hqic": float(getattr(fit, "hqic", float("nan"))),
                        "llf": float(getattr(fit, "llf", float("nan"))),
                    },
                },
            )

        elif model == "sarimax_aic":
            # Auto-order classical baseline (run #2): per-fold AIC search over a
            # small (p,d,q) grid, train-only, fixed seasonal_order=(0,0,0,0)/no
            # trend. Distinct from the fixed-order `sarimax` AR(1) baseline.
            aic_cfg     = cfg["models"].get("SARIMAX_AIC", {})
            p_max       = int(aic_cfg.get("p_max", 2))
            q_max       = int(aic_cfg.get("q_max", 2))
            d_aic       = int(aic_cfg.get("d", 0))
            sarima_aic  = SARIMAModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)

            def _sarima_aic_predict(fit, ti, si, _y=y_series):
                preds, resids, _ = sarima_aic.predict(
                    model_fit=fit, y_test=_y.iloc[si], y_train=_y.iloc[ti])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _y=y_series: sarima_aic.fit_auto(
                    y_train=_y.iloc[ti], p_max=p_max, q_max=q_max, d=d_aic),
                predict_fn=_sarima_aic_predict,
                # last_order is set per fold inside fit_auto and read here right
                # after that fold's fit, so the recorded order is the fold's pick.
                meta_fn=lambda fit: {
                    "params": {"order": list(getattr(sarima_aic, "last_order", ()) or ()),
                               "seasonal_order": list(getattr(sarima_aic, "last_seasonal_order", ()) or ()),
                               "search": {"p_max": p_max, "q_max": q_max, "d": d_aic}},
                    "runtime_seconds": float(getattr(sarima_aic, "runtime_seconds", 0) or 0),
                    "diagnostics": {
                        "aic": float(getattr(fit, "aic", float("nan"))),
                        "bic": float(getattr(fit, "bic", float("nan"))),
                        "chosen_order": list(getattr(sarima_aic, "last_order", ()) or ()),
                    },
                },
            )

        elif model == "varmax":
            varmax_cfg   = cfg["models"]["VARMAX"]
            default_vord = tuple(varmax_cfg.get("order", [1, 0]))
            varmax_mdl   = VARMAXModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)

            def _varmax_predict(fit, ti, si, _y=y_series):
                preds, resids, _ = varmax_mdl.predict(
                    model_fit=fit, y_test=_y.iloc[si], y_train=_y.iloc[ti])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _y=y_series: varmax_mdl.fit(y_train=_y.iloc[ti], o=default_vord),
                predict_fn=_varmax_predict,
                meta_fn=lambda fit: {
                    "params": {"order": list(default_vord)},
                    "runtime_seconds": float(getattr(varmax_mdl, "runtime_seconds", 0) or 0),
                    "diagnostics": {
                        "aic": float(getattr(fit, "aic", float("nan"))),
                        "bic": float(getattr(fit, "bic", float("nan"))),
                        "hqic": float(getattr(fit, "hqic", float("nan"))),
                        "llf": float(getattr(fit, "llf", float("nan"))),
                    },
                },
            )
            n_ok = len(all_results[run_key])
            print(f"[WF] VARMAX: {n_ok}/{n_folds} folds succeeded")

        elif model == "lstm":
            lstm_mdl    = LSTMModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)
            lstm_params = {k: v for k, v in cfg["models"]["LSTM"].items() if k != "ranges"}
            lstm_params["scaler_kind"] = scaler_kind

            def _lstm_predict(fit, ti, si, _X=X_df, _y=y_series):
                preds, resids, _ = lstm_mdl.predict(
                    model_fit=fit,
                    X_train=_X.iloc[ti], X_test=_X.iloc[si],
                    y_train=_y.iloc[ti], y_test=_y.iloc[si])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _X=X_df, _y=y_series: lstm_mdl.fit(
                    X_train=_X.iloc[ti], y_train=_y.iloc[ti], params=lstm_params),
                predict_fn=_lstm_predict,
                meta_fn=lambda fit: {"params": dict(lstm_params),
                                     "runtime_seconds": float(getattr(lstm_mdl, "runtime_seconds", 0) or 0)},
            )

        elif model == "ltc":
            ltc_mdl    = LTCModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)
            ltc_params = {k: v for k, v in cfg["models"]["LTC"].items() if k != "ranges"}
            ltc_params["scaler_kind"] = scaler_kind
            # Run #2 variant overrides (e.g. wiring="fully_connected"). Run-level wins.
            ltc_params.update(run.get("params") or {})

            def _ltc_predict(fit, ti, si, _X=X_df, _y=y_series):
                preds, resids, _ = ltc_mdl.predict(
                    model_fit=fit,
                    X_train=_X.iloc[ti], X_test=_X.iloc[si],
                    y_train=_y.iloc[ti], y_test=_y.iloc[si])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _X=X_df, _y=y_series: ltc_mdl.fit(
                    X_train=_X.iloc[ti], y_train=_y.iloc[ti], params=ltc_params),
                predict_fn=_ltc_predict,
                meta_fn=lambda fit: {"params": dict(ltc_params),
                                     "runtime_seconds": float(getattr(ltc_mdl, "runtime_seconds", 0) or 0)},
            )

        elif model == "cfc":
            cfc_mdl    = CfCModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)
            cfc_params = {k: v for k, v in cfg["models"]["CfC"].items() if k != "ranges"}
            cfc_params["scaler_kind"] = scaler_kind
            # Run #2 variant overrides (mode, wiring, backbone_*). Run-level wins.
            cfc_params.update(run.get("params") or {})

            def _cfc_predict(fit, ti, si, _X=X_df, _y=y_series):
                preds, resids, _ = cfc_mdl.predict(
                    model_fit=fit,
                    X_train=_X.iloc[ti], X_test=_X.iloc[si],
                    y_train=_y.iloc[ti], y_test=_y.iloc[si])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _X=X_df, _y=y_series: cfc_mdl.fit(
                    X_train=_X.iloc[ti], y_train=_y.iloc[ti], params=cfc_params),
                predict_fn=_cfc_predict,
                meta_fn=lambda fit: {"params": dict(cfc_params),
                                     "runtime_seconds": float(getattr(cfc_mdl, "runtime_seconds", 0) or 0)},
            )

        elif model == "ctrnn":
            ctrnn_mdl    = CTRNNModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)
            ctrnn_params = {k: v for k, v in cfg["models"]["CTRNN"].items() if k != "ranges"}
            ctrnn_params["scaler_kind"] = scaler_kind

            def _ctrnn_predict(fit, ti, si, _X=X_df, _y=y_series):
                preds, resids, _ = ctrnn_mdl.predict(
                    model_fit=fit,
                    X_train=_X.iloc[ti], X_test=_X.iloc[si],
                    y_train=_y.iloc[ti], y_test=_y.iloc[si])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _X=X_df, _y=y_series: ctrnn_mdl.fit(
                    X_train=_X.iloc[ti], y_train=_y.iloc[ti], params=ctrnn_params),
                predict_fn=_ctrnn_predict,
                meta_fn=lambda fit: {"params": dict(ctrnn_params),
                                     "runtime_seconds": float(getattr(ctrnn_mdl, "runtime_seconds", 0) or 0)},
            )

        elif model == "neural_ode":
            node_mdl    = NeuralODEModel(cfg=cfg, features=df, logger=fold_sink, ticker=ticker)
            node_params = {k: v for k, v in cfg["models"]["NeuralODE"].items() if k != "ranges"}
            node_params["scaler_kind"] = scaler_kind

            def _node_predict(fit, ti, si, _X=X_df, _y=y_series):
                preds, resids, _ = node_mdl.predict(
                    model_fit=fit,
                    X_train=_X.iloc[ti], X_test=_X.iloc[si],
                    y_train=_y.iloc[ti], y_test=_y.iloc[si])
                return np.asarray(preds, dtype=float), np.asarray(resids, dtype=float)

            all_results[run_key] = _collect_model_folds(
                run_key, folds, y_series, X_df,
                fit_fn=lambda ti, _X=X_df, _y=y_series: node_mdl.fit(
                    X_train=_X.iloc[ti], y_train=_y.iloc[ti], params=node_params),
                predict_fn=_node_predict,
                meta_fn=lambda fit: {"params": dict(node_params),
                                     "runtime_seconds": float(getattr(node_mdl, "runtime_seconds", 0) or 0)},
            )

    # ── Persist + log BEFORE alignment (eng-review issue 9 / Codex #11) ──────────
    # Alignment below can abort (empty primary family / no common folds). Saving
    # completed per-fold forecasts and central logs FIRST means a single failed
    # variant late in an expensive run never discards the work of every other run.
    # Both operate on the full all_results and do not depend on the aligned set.
    _log_folds_centrally(logger, all_results, folds, df_finals, ticker, run_meta, seed,
                         run_identity=run_identity)
    predictions_uri = _persist_predictions(logger, all_results, ticker, run_meta, seed,
                                           run_identity=run_identity)

    # ── Align primary fold set ────────────────────────────────────────────────────
    # Primary family = the runs flagged primary (default: transformed/Condition-1).
    # Derived from run_identity, not a hardcoded list, so run #2's variant family is
    # the primary family without desyncing from the actual run-list. Failed runs
    # (no folds) are excluded with a report rather than aborting the whole run.
    primary_keys = [rk for rk, ident in run_identity.items()
                    if ident["primary"] and all_results.get(rk)]
    excluded = [rk for rk, ident in run_identity.items()
                if ident["primary"] and not all_results.get(rk)]
    if excluded:
        print(f"[WF] Excluded {len(excluded)} primary run(s) with zero successful "
              f"folds from the DM family: {sorted(excluded)}")

    if not primary_keys:
        raise RuntimeError("No primary (Condition-1) runs completed successfully.")

    common_folds = set.intersection(*[set(all_results[k].keys()) for k in primary_keys])
    dropped = n_folds - len(common_folds)
    if dropped > 0:
        print(f"[WF] Dropped {dropped} fold(s) where at least one primary model failed. "
              f"Primary DM tests run on {len(common_folds)} aligned folds.")
    if not common_folds:
        raise RuntimeError("No folds completed successfully across all primary models.")

    # ── Load aligned folds into a single Evaluator ────────────────────────────────
    # DM uses h=1 because per-fold errors are now rolling 1-step-ahead daily
    # errors (both classical and neural). The splitter's `horizon` controls
    # fold size in days, not forecast horizon.
    ev = Evaluator(horizon=1)
    for run_key, fold_results in all_results.items():
        model = run_key.rsplit("_", 1)[0]
        condition = run_key.rsplit("_", 1)[1]
        is_primary = run_key in primary_keys
        # Primary models: only aligned folds; secondary runs: all successful folds
        fold_ids = sorted(common_folds if is_primary else fold_results.keys())
        for fold_id in fold_ids:
            if fold_id not in fold_results:
                continue
            fr = fold_results[fold_id]
            ev.add_fold(run_key, fold_id, fr["y_true"], fr["y_pred"], fr["residuals"])

    # ── Compute metrics ───────────────────────────────────────────────────────────
    metrics = ev.compute()
    # (per-fold logging + prediction snapshot were written before alignment above)

    # ── Primary DM: Condition-1 family, Holm-Bonferroni corrected ────────────────
    primary_sig_table = ev.get_significance_table(primary_models=primary_keys)

    # ── Secondary DM: C1 vs C2 per neural model, uncorrected ─────────────────────
    neural_models   = ("lstm", "ltc", "cfc", "ctrnn", "neural_ode")
    secondary_rows  = []
    for nm in neural_models:
        k1, k2 = f"{nm}_transformed", f"{nm}_raw"
        if k1 not in ev._folds or k2 not in ev._folds:
            continue
        e1 = np.concatenate([f["y_pred"] - f["y_true"] for f in ev._folds[k1]])
        e2 = np.concatenate([f["y_pred"] - f["y_true"] for f in ev._folds[k2]])
        stat, p_val = dm_test_stat(e1, e2, h=1)
        secondary_rows.append({"model": nm, "key_transformed": k1, "key_raw": k2,
                                "dm_stat": stat, "p_raw": p_val})
    secondary_sig_table = pd.DataFrame(secondary_rows) if secondary_rows else pd.DataFrame(
        columns=["model", "key_transformed", "key_raw", "dm_stat", "p_raw"])

    # ── Print summary ─────────────────────────────────────────────────────────────
    print("\n=== Walk-Forward Metrics ===")
    for m, v in metrics.items():
        print(f"  {m:30s}: RMSE={v['rmse_mean']:.6f} ±{v['rmse_std']:.6f}  "
              f"MAE={v['mae_mean']:.6f}  DA={v['directional_accuracy_mean']:.3f}  "
              f"folds={v['n_folds']}")

    n_pairs = len(primary_keys) * (len(primary_keys) - 1) // 2
    print(f"\n=== Primary DM Table ({len(primary_keys)} primary runs, "
          f"Holm-Bonferroni corrected, {n_pairs} pairs) ===")
    print(primary_sig_table.to_string(index=False))

    if not secondary_sig_table.empty:
        print("\n=== Secondary DM Table (C1 vs C2 per neural model, uncorrected) ===")
        print(secondary_sig_table.to_string(index=False))

    # ── Residual diagnostics ──────────────────────────────────────────────────────
    diagnostics: dict = {}
    for run_key in metrics:
        try:
            diagnostics[run_key] = ev.check_residuals(run_key)
        except ValueError:
            diagnostics[run_key] = {"note": "residuals not logged for all folds"}

    # ── Persist to disk ───────────────────────────────────────────────────────────
    summary = {
        "ticker": ticker,
        "n_folds_total": n_folds,
        "n_folds_primary_aligned": len(common_folds),
        "initial_window": initial_window,
        "horizon": horizon,
        "seed": seed,
        "metrics": metrics,
        "primary_significance_table": primary_sig_table.to_dict(orient="records"),
        "secondary_significance_table": secondary_sig_table.to_dict(orient="records"),
        "diagnostics": diagnostics,
    }
    save_results(output_dir=output_dir, summary=summary,
                 predictions_df=None, filename_prefix=f"{ticker}_wf")
    print(f"\n[WF] Results saved to {output_dir}/{ticker}_wf_summary.json")

    # ── Persist to Supabase (experiment_summaries) ───────────────────────────────
    # Mirrors the on-disk summary into Supabase so cloud VM teardown doesn't lose
    # the headline numbers. Also bundles any rows that landed in the resilient
    # logger's local JSONL fallback during this run (paused-Supabase recovery).
    run_meta = cfg.get("_run_meta", {})
    # ResilientLogger writes fallbacks to "results/logger_fallback" by default;
    # derive that from output_dir which sits next to it under "results/".
    fallback_dir = os.path.join(os.path.dirname(output_dir), "logger_fallback")
    from provenance import env_manifest as _env_manifest
    # Resolved config minus the volatile _run_meta (kept separately as env_manifest).
    config_snapshot = {k: v for k, v in cfg.items() if k != "_run_meta"}
    summary_row = {
        "ticker": ticker,
        "run_id": run_meta.get("run_id"),
        "run_group_id": run_meta.get("run_group_id"),
        "git_commit": run_meta.get("git_commit"),
        "n_folds_total": int(n_folds),
        "n_folds_primary_aligned": int(len(common_folds)),
        "initial_window": int(initial_window),
        "horizon": int(horizon),
        "seed": int(seed),
        "metrics": metrics,
        "primary_significance_table": primary_sig_table.to_dict(orient="records"),
        "secondary_significance_table": secondary_sig_table.to_dict(orient="records"),
        "diagnostics": diagnostics,
        "summary_full": summary,
        # ── Provenance (Workstream A) ──────────────────────────────────────
        "config_snapshot": config_snapshot,
        "env_manifest": _env_manifest(seed=seed, run_id=run_meta.get("run_id"),
                                      run_group_id=run_meta.get("run_group_id")),
        "data_fingerprint": input_fingerprints,
        "snapshot_uri": {"inputs": snapshot_uris, "predictions": predictions_uri},
        "fallback_contents": _read_fallback_jsonl(
            fallback_dir,
            since_ts=run_meta.get("run_start_ts"),
            ticker=ticker,
            run_id=run_meta.get("run_id"),
        ),
    }
    try:
        logger.log_summary(summary_row)
    except Exception as exc:
        # log_summary on ResilientLogger should never raise, but guard anyway so
        # the cloud run never dies on summary upload after the fold work is done.
        print(f"[WF] log_summary raised unexpectedly for {ticker}: {exc}")

    return metrics, primary_sig_table, secondary_sig_table


def _read_fallback_jsonl(fallback_dir: str, since_ts: Optional[float] = None,
                         ticker: Optional[str] = None,
                         run_id: Optional[str] = None) -> dict:
    """Read *.jsonl files in fallback_dir and return {table_name: [rows]}.

    Filters:
    - since_ts: only rows whose `_fallback_ts` (epoch seconds, written by
      ResilientLogger) is >= since_ts. Rows without `_fallback_ts` are dropped
      when since_ts is set (cannot be attributed to this run).
    - ticker: only rows whose `ticker` field matches. Rows without a `ticker`
      field are kept (e.g., summary fallbacks).
    - run_id: only rows whose `run_id` field matches this sub-run. Rows without
      a `run_id` field are kept. Without this, a multi-seed run-group shares one
      `run_start_ts`, so a later seed's summary would absorb earlier seeds'
      fallback rows for the same ticker and pollute its per-(ticker, seed)
      provenance.

    Returns empty dict if the directory does not exist or contains no rows
    matching the filters. Never raises — fallback collection is best-effort.
    """
    out: dict = {}
    if not os.path.isdir(fallback_dir):
        return out
    for path in glob.glob(os.path.join(fallback_dir, "*.jsonl")):
        table = os.path.splitext(os.path.basename(path))[0]
        rows = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        rows.append({"_raw": line})
                        continue
                    if since_ts is not None:
                        row_ts = row.get("_fallback_ts")
                        if row_ts is None or float(row_ts) < since_ts:
                            continue
                    if ticker is not None and "ticker" in row and row["ticker"] != ticker:
                        continue
                    if run_id is not None and "run_id" in row and row["run_id"] != run_id:
                        continue
                    rows.append(row)
        except Exception as exc:
            rows = [{"_read_error": str(exc)}]
        if rows:
            out[table] = rows
    return out


# ─── RUNNING THE ENTIRE EXPERIMENT ────────────────────────────────────────────────────────────────────────────────────

def run_experiment(ticker: str, period: str, interval: str, cfg, logger):

    """
    Runs, controls, and manages the bachelor thesis experiment in its entirety.

    :param ticker: Symbol of stock being currently analysed, e.g. "AAPL".
    :param period: Period for which data should be fetched, e.g. "5y" or "2mo".
    :param interval: Interval for which data should be fetched, e.g. "1d" or "1mo".
    :param cfg: Your config.yaml file.
    :param logger: Shared ResilientLogger built once in main() (so the crash-resume
        check and the per-ticker runs use the same client).
    """

    # 1-3) LOAD + CLEAN DATA — only on the live seed. Seeds 2..N of a run-group
    #    load the exact input frames from the seed-1 snapshot (eng-review I3), so
    #    re-fetching live here would be wasted work AND risk vendor drift /
    #    rate-limits. The walk-forward resolver ignores these dicts under
    #    data_source=snapshot, so empty dicts are safe.
    data_source = (cfg.get("_run_meta") or {}).get("data_source", "live")
    if data_source == "snapshot":
        clean_internal, clean_external = {}, {}
    else:
        raw_internal = get_internal_data(ticker=ticker, period=period, interval=interval, cfg=cfg)
        raw_external = get_external_data(period=period, interval=interval, cfg=cfg)
        clean_internal, clean_external = get_clean_data(
            raw_internal=raw_internal, raw_external=raw_external, cfg=cfg)


    # 4-6) LEGACY SINGLE-SHOT PATH — DISABLED 2026-06-08 (commented, not deleted).
    #    get_features -> get_compliance_check -> the single-shot model fits below
    #    are superseded by the walk-forward evaluation (step 7). They are dead
    #    weight: nothing downstream consumes their output, they double the neural
    #    training time, and get_compliance_check writes df_final to a hardcoded
    #    macOS Desktop path (get_compliance_check, ~main.py:270) that would crash
    #    on a Linux cloud VM. The walk-forward path builds its own per-condition
    #    feature frames internally, so none of this is needed.
    #
    # df = get_features(clean_internal=clean_internal, clean_external=clean_external, cfg=cfg)
    # df_final = get_compliance_check(df)
    # get_sarimax(df_final=df_final, logger=logger, ticker=ticker, cfg=cfg)
    # get_varmax(df_final=df_final, logger=logger, ticker=ticker, cfg=cfg)
    # get_lstm(df_final=df_final, logger=logger, ticker=ticker, cfg=cfg)
    # get_ltc(df_final=df_final, logger=logger, ticker=ticker, cfg=cfg)
    # get_cfc(df_final=df_final, logger=logger, ticker=ticker, cfg=cfg)

    # 7) WALK-FORWARD EVALUATION
    #    Logger is built once in main() (wraps SupabaseLogger in ResilientLogger so a
    #    paused/free-tier Supabase or transient network failure writes to a local
    #    JSONL fallback instead of killing the grid) and passed in here.
    #    Dual-condition run over all experiment.runs; builds df_finals per condition internally
    run_walk_forward_evaluation(
        clean_internal=clean_internal,
        clean_external=clean_external,
        logger=logger,
        ticker=ticker,
        cfg=cfg,
    )



def main(config_path="config.yaml"):

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Run #2b stationarized-exog arm (approach B). When the config sets
    # experiment.exog_stationarize, override every exogenous feature's transform
    # per features/stationarize.py EXOG_TRANSFORM_TABLE (the single source of
    # truth) right after load — before validation, snapshotting, and any seed —
    # so the whole pipeline sees properly-conditioned inputs. The transform
    # decisions live ONLY in the Python table; config.run2b_stationary.yaml is
    # config.yaml plus this one flag (see scripts/gen_run2b_config.py). Unset /
    # false → raw-level run #2 behaviour, unchanged. Raises if any exog feature
    # lacks a reviewed table entry (fail-fast before any writes).
    if cfg.get("experiment", {}).get("exog_stationarize", False):
        cfg = stationarize_exog_cfg(cfg)
        print("[run] exog_stationarize=ON: exogenous transforms overridden per "
              "EXOG_TRANSFORM_TABLE (run #2b stationarized arm).")

    validate_experiment_config(cfg)
    assert_run_kind(cfg, config_path)

    # Run-wide metadata shared across all tickers in this main() invocation.
    # git_commit is ENFORCED (eng-review A1): a run with no provenance commit is
    # not reproducible. Prefer an explicit GIT_COMMIT env (the deploy script sets
    # it before scp, since the box is not a git checkout); else read it from git;
    # else abort, unless ALLOW_NO_GIT_COMMIT=1 is set for local/dev throwaway runs.
    git_commit = _env("GIT_COMMIT")
    if not git_commit:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_commit = ""
    if not git_commit and not _env("ALLOW_NO_GIT_COMMIT"):
        raise RuntimeError(
            "git_commit could not be determined (not a git checkout and GIT_COMMIT "
            "unset). Provenance requires a commit. Fix: run from a git checkout, or "
            "set GIT_COMMIT=$(git rev-parse HEAD) before launch (the deploy script "
            "does this), or set ALLOW_NO_GIT_COMMIT=1 to override for a throwaway run."
        )
    # run_id groups all per-ticker summary rows of one experiment. It can be
    # pinned via the RUN_ID env var so a CRASHED RUN RESUMES: re-launch with the
    # same RUN_ID and tickers whose summary row already exists are skipped
    # (see the per-ticker check below). Unset → fresh id, so a normal launch
    # never accidentally resumes a prior run.
    run_id = _env("RUN_ID")
    if not run_id:
        # Auto-generated: second-level UTC timestamp + short random suffix so two
        # launches in the same second don't collide on run_id. An explicit
        # RUN_ID (the resume path) is honored verbatim and never suffixed.
        run_id = (datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                  + "-" + uuid.uuid4().hex[:6])
    # run_id is the RUN-GROUP id: all per-seed sub-runs of this experiment share
    # it (per-seed run_id = "{run_group_id}#s{seed}"). The input snapshot is
    # stored once per group (seed-1 writes, later seeds load — eng-review I3).
    run_group_id = run_id
    run_start_ts = time.time()
    print(f"\n[run] run_group_id={run_group_id} git_commit={git_commit[:8] or 'unknown'}")
    print(f"[run] to resume after a crash, re-launch with RUN_ID={run_group_id} "
          "(completed (ticker, seed) pairs are skipped).")

    # Build the logger ONCE (was per-ticker). Wraps SupabaseLogger in
    # ResilientLogger; also used for the crash-resume existence check below.
    if not (SUPABASE_URL and SUPABASE_KEY):
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_KEY are not set. Copy .env.example to .env and "
            "fill them in (or pass --env-file .env to docker run)."
        )
    logger = ResilientLogger(SupabaseLogger(SUPABASE_URL, SUPABASE_KEY))

    tickers = cfg["data"]["internal"]["tickers"]
    scenarios = cfg["experiment"]["scenarios"]
    # Multi-seed: run the full grid once per seed as a sub-run. Classical models
    # are deterministic, so they are re-run each seed (identical results) to keep
    # every seed's DM table self-contained; aggregation treats them as
    # seed-invariant. (This is the simple, correct alternative to sharing
    # classical predictions across sub-run processes — see PLAN open-question.)
    seeds = cfg["experiment"].get("seeds") or [int(cfg["experiment"].get("seed", 42))]
    seeds = [int(s) for s in seeds]
    multi_seed = len(seeds) > 1
    # Input pinning (run #1/#2): when INPUT_GROUP_ID is set, EVERY seed loads the
    # frozen input snapshot from that group instead of rebuilding live. This keeps
    # the inputs byte-identical to a prior run (e.g. #0 = grid-2026-06-11) so the
    # only thing that changed is the model/feature config. Results still write
    # under this run's own run_group_id (RUN_ID). Unset → normal live build.
    input_group_id = _env("INPUT_GROUP_ID") or None
    assert_stationarize_input_source(cfg, input_group_id)  # review P0 (run #2b)
    if input_group_id:
        print(f"[run] INPUT pinned to group '{input_group_id}': all seeds load its "
              "frozen input snapshot (no live rebuild); results write under "
              f"run_group_id={run_group_id}.")
    elif multi_seed:
        print(f"[run] multi-seed {seeds}: seed {seeds[0]} builds + uploads the input "
              "snapshot; later seeds load it (identical inputs across seeds).")

    # Crash-resume keys on (run_id, ticker) only — experiment_summaries has no
    # scenario field. With >1 scenario the same ticker recurs across scenarios
    # and that key would collide, so the skip is only safe for a single
    # scenario. With multiple scenarios resume is disabled (all tickers run);
    # adding a scenario column to experiment_summaries would lift this.
    # Note: the per-ticker completion marker lives in Supabase. If Supabase is
    # down, log_summary falls back to a local JSONL that does not survive a
    # `docker run --rm` restart anyway, so summary_exists fails open (returns
    # False) and the ticker is simply re-run — the correct behaviour.
    resume_enabled = _resume_enabled(scenarios, _env("RUN_MODELS"))
    if not resume_enabled:
        if _env("RUN_MODELS"):
            print("[run] RUN_MODELS set (filtered family batch): per-(ticker,seed) "
                  "summary-resume DISABLED so a partial-family summary can't mark a "
                  "ticker complete. Use a DISTINCT RUN_ID per family batch.")
        else:
            print(f"[run] {len(scenarios)} scenarios configured; per-ticker crash-resume "
                  "is disabled (needs a scenario field in experiment_summaries).")

    for scenario in scenarios:

        name = scenario["name"]
        period = scenario["period"]
        interval = scenario["interval"]

        print(f"\n=== Scenario: {name} ({period} @ {interval}) ===")

        for ticker in tickers:
            for si, seed in enumerate(seeds):
                # Per-seed sub-run id. Seed 1 builds from live data + uploads the
                # group input snapshot; seeds 2..N load it (data_source=snapshot).
                seed_run_id = f"{run_group_id}#s{seed}" if multi_seed else run_group_id
                # Pinned-input runs read a frozen snapshot for EVERY seed (nothing
                # is built live, so there is no seed-1 upload). Otherwise the
                # normal handoff: seed 1 builds live + uploads, seeds 2..N load it.
                if input_group_id:
                    data_source = "snapshot"
                else:
                    data_source = "live" if si == 0 else "snapshot"
                cfg["_run_meta"] = {
                    "run_id": seed_run_id,
                    "run_group_id": run_group_id,
                    "input_group_id": input_group_id,
                    "git_commit": git_commit,
                    "run_start_ts": run_start_ts,
                    "data_source": data_source,
                    "seed": seed,
                    "multi_seed": multi_seed,
                }

                # Crash-resume: skip a (ticker, seed) whose summary already
                # exists for this sub-run id (summary is written LAST).
                if resume_enabled and logger.summary_exists(seed_run_id, ticker):
                    print(f"\n--- Skipping {ticker} seed={seed}: summary already in "
                          f"Supabase for run_id={seed_run_id} (resume). ---")
                    continue

                print(f"\n--- Running ticker={ticker} seed={seed} ({data_source}) ---")
                run_experiment(ticker, period, interval, cfg, logger)



if __name__ == "__main__":
    import sys
    # Optional config path override: `python main.py config.smoke.yaml`.
    # Defaults to config.yaml (the full grid) when no arg is given.
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
