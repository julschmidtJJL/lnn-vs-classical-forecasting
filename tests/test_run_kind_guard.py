"""
Tests for the RUN_KIND startup guard (main.assert_run_kind).

The repo has three run configs selected ONLY by the positional CLI arg, with
dangerous defaults (bare `python main.py` -> config.yaml = run #2; Docker CMD ->
smoke). The guard stops a wrong config from being launched under a named RUN_ID:

- RUN_KIND set -> must equal the config's self-declared experiment.run_kind.
- RUN_KIND unset but RUN_ID set -> abort (a named/resumable run must declare its
  kind, else `RUN_ID=run1 python main.py` silently loads run #2 under #1's group).
- both unset -> allowed (local/smoke throwaway).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import main


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("RUN_KIND", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)


def _cfg(run_kind):
    return {"experiment": ({"run_kind": run_kind} if run_kind is not None else {})}


def test_both_unset_allows_local_run():
    # No RUN_ID, no RUN_KIND -> local throwaway, guard is print-only.
    main.assert_run_kind(_cfg("run2"), "config.yaml")


def test_named_run_without_run_kind_aborts(monkeypatch):
    # The P1 case: RUN_ID set but RUN_KIND unset would silently load the default
    # config.yaml (run #2) under a run #1 RUN_ID. Must abort.
    monkeypatch.setenv("RUN_ID", "run1-logreturn-2026-06-18")
    with pytest.raises(RuntimeError, match="RUN_ID is set"):
        main.assert_run_kind(_cfg("run2"), "config.yaml")


def test_matching_run_kind_passes(monkeypatch):
    monkeypatch.setenv("RUN_ID", "run1-x")
    monkeypatch.setenv("RUN_KIND", "run1")
    main.assert_run_kind(_cfg("run1"), "config.run1.yaml")


def test_mismatched_run_kind_aborts(monkeypatch):
    monkeypatch.setenv("RUN_ID", "run1-x")
    monkeypatch.setenv("RUN_KIND", "run2")
    with pytest.raises(RuntimeError, match="RUN_KIND mismatch"):
        main.assert_run_kind(_cfg("run1"), "config.run1.yaml")


def test_run_kind_set_but_config_undeclared_aborts(monkeypatch):
    monkeypatch.setenv("RUN_KIND", "run1")
    with pytest.raises(RuntimeError, match="declares no"):
        main.assert_run_kind(_cfg(None), "config.weird.yaml")


def test_invalid_declared_kind_rejected():
    with pytest.raises(ValueError, match="not one of"):
        main.assert_run_kind(_cfg("run9"), "config.bad.yaml")


@pytest.mark.parametrize("path,kind", [
    ("config.yaml", "run2"),
    ("config.run1.yaml", "run1"),
    ("config.smoke.yaml", "smoke"),
])
def test_shipped_configs_declare_valid_kind(path, kind):
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "src", path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["experiment"]["run_kind"] == kind
