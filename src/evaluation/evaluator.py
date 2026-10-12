import itertools
import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.multitest import multipletests



def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(float).reshape(-1)
    y_pred = np.asarray(y_pred).astype(float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(float).reshape(-1)
    y_pred = np.asarray(y_pred).astype(float).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """MAPE in percent. Uses epsilon guard for near-zero actuals."""
    y_true = np.asarray(y_true).astype(float).reshape(-1)
    y_pred = np.asarray(y_pred).astype(float).reshape(-1)
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of periods where predicted sign matches actual sign."""
    y_true = np.asarray(y_true).astype(float).reshape(-1)
    y_pred = np.asarray(y_pred).astype(float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}"
        )
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def dm_test_stat(e1, e2, h: int = 1) -> Tuple[float, float]:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction.

    e1, e2: forecast error arrays (actual - predicted), shape (T,).
    Positive DM stat → model 2 has larger loss → model 1 is better.
    Returns (dm_stat, p_value) using t(T-1) distribution.
    """
    e1 = np.asarray(e1, dtype=float).reshape(-1)
    e2 = np.asarray(e2, dtype=float).reshape(-1)
    T = len(e1)

    # squared-error loss differential
    d = e1 ** 2 - e2 ** 2
    d_bar = np.mean(d)

    # Newey-West HAC variance of d_bar: bandwidth = h-1 lags
    gamma_0 = np.mean((d - d_bar) ** 2)
    nw_sum = gamma_0
    for lag in range(1, h):
        w = 1.0 - lag / h  # Bartlett weight
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        nw_sum += 2.0 * w * gamma_l

    var_d_bar = nw_sum / T
    if var_d_bar <= 0.0:
        return 0.0, 1.0

    dm_raw = d_bar / np.sqrt(var_d_bar)

    # HLN correction factor (degrades gracefully when term ≤ 0)
    hln_term = T + 1 - 2 * h + h * (h - 1) / T
    dm_stat = dm_raw * np.sqrt(hln_term / T) if hln_term > 0 else dm_raw

    p_value = float(2.0 * (1.0 - scipy_stats.t.cdf(abs(dm_stat), df=T - 1)))
    return float(dm_stat), p_value


def apply_holm_bonferroni(
    p_values: Dict[str, float], alpha: float = 0.05
) -> Dict[str, Dict]:
    """
    Holm-Bonferroni correction for a dict of p-values.
    Returns {key: {p_raw, p_corrected, significant}} for each key.
    """
    if not p_values:
        return {}

    keys = list(p_values.keys())
    pvals = np.array([p_values[k] for k in keys])
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    return {
        keys[i]: {
            "p_raw": float(pvals[i]),
            "p_corrected": float(pvals_corrected[i]),
            "significant": bool(reject[i]),
        }
        for i in range(len(keys))
    }


DEFAULT_METRICS: Dict[str, Callable] = {"rmse": rmse, "mae": mae, "mape": mape}


def evaluate_predictions(
    y_true, y_pred, metrics: Optional[Iterable[Callable]] = None
) -> Dict[str, float]:
    """
    Compute a set of metrics for predicted vs true values.

    metrics: None (uses DEFAULT_METRICS) | iterable of callables | dict name→callable.
    Returns {metric_name: score, "n": n_observations}.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    if not isinstance(metrics, dict):
        metrics = {getattr(fn, "__name__", str(fn)): fn for fn in metrics}

    out: Dict[str, float] = {}
    for name, fn in metrics.items():
        try:
            out[name] = float(fn(y_true, y_pred))
        except (TypeError, ValueError):
            out[name] = float("nan")

    out["n"] = int(len(y_true))
    return out


def save_results(
    output_dir: str,
    summary: Dict,
    predictions_df: Optional[pd.DataFrame] = None,
    filename_prefix: str = "loggers",
):
    """Persist experiment summary and optional predictions to disk."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"{filename_prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if predictions_df is not None:
        predictions_df.to_csv(
            os.path.join(output_dir, f"{filename_prefix}_predictions.csv")
        )


class Evaluator:
    """
    Walk-forward evaluation accumulator.

    Usage:
        ev = Evaluator(horizon=21)
        for fold_id, (y_true, y_pred, residuals) in enumerate(folds):
            ev.add_fold("sarimax", fold_id, y_true, y_pred, residuals)
        metrics = ev.compute()
        sig_table = ev.get_significance_table()
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        # model_key → list of fold dicts
        self._folds: Dict[str, List[Dict]] = {}

    def add_fold(
        self,
        model_key: str,
        fold_id: int,
        y_true,
        y_pred,
        residuals=None,
        train_start=None,
        train_end=None,
        test_start=None,
        test_end=None,
    ) -> None:
        """Accumulate one walk-forward fold for model_key."""
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true/y_pred length mismatch in fold {fold_id}: "
                f"{len(y_true)} vs {len(y_pred)}"
            )

        if residuals is not None:
            residuals = np.asarray(residuals, dtype=float).reshape(-1)
            if len(residuals) != len(y_true):
                raise ValueError(
                    f"residuals length mismatch in fold {fold_id}: "
                    f"{len(residuals)} vs {len(y_true)}"
                )

        if model_key not in self._folds:
            self._folds[model_key] = []

        self._folds[model_key].append(
            {
                "fold_id": fold_id,
                "y_true": y_true,
                "y_pred": y_pred,
                "residuals": residuals,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

    def compute(self) -> Dict[str, Dict]:
        """
        Return mean ± std of RMSE, MAE, and directional accuracy across all folds
        for every registered model.
        """
        results: Dict[str, Dict] = {}

        for model_key, folds in self._folds.items():
            fold_rmse = [rmse(f["y_true"], f["y_pred"]) for f in folds]
            fold_mae = [mae(f["y_true"], f["y_pred"]) for f in folds]
            fold_da = [directional_accuracy(f["y_true"], f["y_pred"]) for f in folds]

            results[model_key] = {
                "rmse_mean": float(np.mean(fold_rmse)),
                "rmse_std": float(np.std(fold_rmse, ddof=0)),
                "mae_mean": float(np.mean(fold_mae)),
                "mae_std": float(np.std(fold_mae, ddof=0)),
                "directional_accuracy_mean": float(np.mean(fold_da)),
                "directional_accuracy_std": float(np.std(fold_da, ddof=0)),
                "n_folds": len(folds),
            }

        return results

    def get_significance_table(self, primary_models: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Pairwise DM tests with Holm-Bonferroni correction.

        primary_models: restrict the corrected family to this list of model keys.
            None → all registered models (old behavior).
            Provide the 6 Condition-1 model keys to get the correct 15-pair family.

        Returns DataFrame: model_a | model_b | dm_stat | p_raw | p_corrected | significant
        """
        models = primary_models if primary_models is not None else list(self._folds.keys())
        # Only keep models that actually have fold data
        models = [m for m in models if m in self._folds]
        pairs = list(itertools.combinations(models, 2))

        if not pairs:
            return pd.DataFrame(
                columns=["model_a", "model_b", "dm_stat", "p_raw", "p_corrected", "significant"]
            )

        rows = []
        for m_a, m_b in pairs:
            # concatenate forecast errors across folds
            e_a = np.concatenate(
                [f["y_pred"] - f["y_true"] for f in self._folds[m_a]]
            )
            e_b = np.concatenate(
                [f["y_pred"] - f["y_true"] for f in self._folds[m_b]]
            )
            stat, p_val = dm_test_stat(e_a, e_b, h=self.horizon)
            pair_key = f"{m_a}_vs_{m_b}"
            rows.append({"pair_key": pair_key, "model_a": m_a, "model_b": m_b,
                         "dm_stat": stat, "p_raw": p_val})

        df = pd.DataFrame(rows)

        corrected = apply_holm_bonferroni(
            {r["pair_key"]: r["p_raw"] for r in rows}
        )
        df["p_corrected"] = df["pair_key"].map(
            lambda k: corrected[k]["p_corrected"]
        )
        df["significant"] = df["pair_key"].map(
            lambda k: corrected[k]["significant"]
        )
        return df.drop(columns=["pair_key"]).reset_index(drop=True)

    def friedman_cross_asset(
        self,
        rmses_by_asset: Dict[str, Dict[str, float]],
        models: Optional[List[str]] = None,
    ) -> Dict:
        """
        Friedman rank test on per-asset model RMSE ranks (treatments = models,
        blocks = assets). Use as the cross-sectional statistic for the 3-asset
        study.

        Per the codex review (D4-codex), this is reported as **descriptive
        evidence of rank consistency**, not a confirmatory test — n=3 blocks
        gives weak power and the asymptotic χ² approximation is unreliable.
        No Nemenyi post-hoc.

        Args:
            rmses_by_asset: {asset_key: {model_key: rmse_value}}. Every asset
                must have an RMSE for every model in ``models`` (after the
                optional filter); ragged input raises.
            models: Restrict the comparison to this model list. ``None`` →
                intersection of model keys across all assets.

        Returns:
            {
                "assets": [...],          # asset order used
                "models": [...],          # model order used
                "ranks_by_asset": {asset: [rank_for_model_0, rank_for_model_1, ...]},
                "chi2": float,
                "p_value": float,
                "n_assets": int,
                "n_models": int,
                "note": "descriptive; n_assets=N is weak — no Nemenyi post-hoc",
            }
        """
        assets = list(rmses_by_asset.keys())
        if len(assets) < 2:
            raise ValueError(
                f"Friedman requires ≥ 2 assets (blocks), got {len(assets)}."
            )

        if models is None:
            common = set(rmses_by_asset[assets[0]].keys())
            for a in assets[1:]:
                common &= set(rmses_by_asset[a].keys())
            models = sorted(common)

        if len(models) < 3:
            raise ValueError(
                f"Friedman requires ≥ 3 treatments (models), got {len(models)}. "
                "scipy.stats.friedmanchisquare degenerates below 3."
            )

        # Per-asset rank vectors (rankdata: lower RMSE → rank 1, ties averaged).
        ranks_by_asset: Dict[str, List[float]] = {}
        for a in assets:
            rmses = []
            for m in models:
                if m not in rmses_by_asset[a]:
                    raise ValueError(
                        f"Missing RMSE for ({a}, {m}); Friedman input must be "
                        "rectangular."
                    )
                rmses.append(float(rmses_by_asset[a][m]))
            ranks_by_asset[a] = scipy_stats.rankdata(rmses).tolist()

        # scipy.friedmanchisquare expects one array per TREATMENT (model), each
        # of length n_blocks (assets). Transpose ranks_by_asset accordingly.
        per_model_ranks = [
            [ranks_by_asset[a][i] for a in assets] for i in range(len(models))
        ]
        chi2, p_value = scipy_stats.friedmanchisquare(*per_model_ranks)

        return {
            "assets": assets,
            "models": models,
            "ranks_by_asset": ranks_by_asset,
            "chi2": float(chi2),
            "p_value": float(p_value),
            "n_assets": len(assets),
            "n_models": len(models),
            "note": (
                f"descriptive; n_assets={len(assets)} is weak — "
                "no Nemenyi post-hoc"
            ),
        }

    def check_residuals(self, model_key: str) -> Dict:
        """
        ARCH and Ljung-Box diagnostic on concatenated fold residuals.

        Returns {arch_stat, arch_p, ljung_box_p, heteroscedastic}.
        Heteroscedastic = True if ARCH p-value < 0.05 (triggers HLN note in thesis).
        """
        residual_arrays = [
            f["residuals"]
            for f in self._folds.get(model_key, [])
            if f["residuals"] is not None
        ]

        if not residual_arrays:
            raise ValueError(
                f"No residuals found for model '{model_key}'. "
                "Pass residuals= in add_fold() to enable this check."
            )

        all_residuals = np.concatenate(residual_arrays)

        # Drop non-finite residuals before the diagnostics. A model that
        # diverged (e.g. NaN training loss -> NaN predictions/residuals) would
        # otherwise make het_arch/OLS raise MissingDataError ("exog contains
        # inf or nans") and crash the whole walk-forward run BEFORE the
        # experiment_summaries logging. One broken model must not take the run
        # down; skip its diagnostic gracefully instead. Healthy models have no
        # non-finite residuals, so their results are unchanged.
        finite_residuals = all_residuals[np.isfinite(all_residuals)]
        n_dropped = int(all_residuals.size - finite_residuals.size)

        # het_arch fits an OLS on lagged squared residuals and Ljung-Box uses
        # lag 10; both need a non-trivial finite sample to be meaningful.
        if finite_residuals.size < 20:
            return {
                "arch_stat": float("nan"),
                "arch_p": float("nan"),
                "ljung_box_p": float("nan"),
                "heteroscedastic": False,
                "note": (
                    f"residual diagnostics skipped: only {finite_residuals.size} "
                    f"finite residuals ({n_dropped} non-finite dropped)"
                ),
            }

        arch_stat, arch_p, _, _ = het_arch(finite_residuals)
        lb_result = acorr_ljungbox(finite_residuals, lags=[10], return_df=True)
        lb_p = float(lb_result["lb_pvalue"].iloc[0])

        result = {
            "arch_stat": float(arch_stat),
            "arch_p": float(arch_p),
            "ljung_box_p": lb_p,
            "heteroscedastic": bool(arch_p < 0.05),
        }
        if n_dropped:
            result["note"] = f"{n_dropped} non-finite residual(s) dropped before diagnostics"
        return result
