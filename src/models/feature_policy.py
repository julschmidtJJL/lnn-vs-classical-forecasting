"""Single source of truth for whether the past target (``log_return``) is fed
to the neural models as an input feature.

Background: in the windowed setup each neural model builds
``X[i:i+seq_len] -> y[i+seq_len]``, so the window ends at row ``t-1`` when
predicting ``y_t``. Including ``log_return`` in ``X`` therefore only exposes
PAST returns — legitimate autoregression, the same information AR(1) already
uses. It is NOT look-ahead leakage under the rolling 1-step-ahead walk-forward
contract.

Run #0 (``grid-2026-06-11``) dropped ``log_return`` from the neural inputs
entirely (the "before"/handicapped baseline). Run #1 keeps it by setting
``experiment.include_past_target_as_feature: true``. The flag is consulted in
two places — the orchestrator's per-run X build (``main.py``) and each model's
``fit()`` — so this module exists to keep those two layers in agreement and
make it impossible to silently gate one but not the other.
"""
from __future__ import annotations

TARGET_COL = "log_return"


def include_past_target(cfg) -> bool:
    """Return True when the past target column should be kept as an input feature.

    Defaults to False (drop it) so behaviour matches run #0 unless a config
    explicitly opts in. Tolerates ``cfg`` being None or a non-dict.
    """
    try:
        return bool((cfg or {}).get("experiment", {}).get(
            "include_past_target_as_feature", False))
    except AttributeError:
        return False


def apply_target_policy(X, cfg):
    """Drop the target column from feature frame ``X`` unless the flag keeps it.

    Returns ``X`` with ``log_return`` removed when the flag is off (run #0
    behaviour), or untouched when the flag is on (run #1). The caller is
    responsible for validating that at least one feature column remains.
    """
    if X is None:
        return X
    if include_past_target(cfg):
        return X
    if hasattr(X, "columns") and TARGET_COL in X.columns:
        return X.drop(columns=[TARGET_COL])
    return X
