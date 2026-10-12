#!/usr/bin/env python3
"""
Generate ``src/config.run2b_stationary.yaml`` — the run #2b stationarized-exog arm.

Approach B (separate config + RUN_ID, see docs/run2_plan.md): the stationarized
arm is config.yaml with two changes — ``experiment.run_kind`` set to ``run2b``
and the flag ``experiment.exog_stationarize: true``. main.py applies
features/stationarize.py::stationarize_exog_cfg at load time when that flag is
set, so the per-feature transform DECISIONS live only in the Python table (the
single source of truth) — this generated YAML carries no transform values.

The distinct ``run_kind: run2b`` makes assert_run_kind hard-refuse launching this
config under run #2's ``RUN_KIND``/``RUN_ID``, so the two arms can never silently
mix into one result group.

This generator does a TEXT copy and, at the ``run_kind: run2`` line inside the
``experiment:`` block, rewrites the value to ``run2b`` and injects the flag right
after. That preserves every comment in the ~1500-line config and means re-running
picks up any later edits to config.yaml (FRED series, seeds, run-list) with zero
drift. Re-run after any change to config.yaml.

Launch — INPUT_GROUP_ID MUST be unset. The stationarizing transform only applies
when inputs are built live; pinning to a pre-built group (e.g. grid-2026-06-11)
would silently train on non-stationarized inputs. main.py hard-refuses the combo
(exog_stationarize + INPUT_GROUP_ID) unless ALLOW_PINNED_STATIONARIZED=1. This arm
builds and snapshots its own stationarized inputs under its own RUN_ID:

    RUN_KIND=run2b RUN_ID=<new> python src/main.py config.run2b_stationary.yaml
"""

from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
_IN = os.path.join(_SRC, "config.yaml")
_OUT = os.path.join(_SRC, "config.run2b_stationary.yaml")
# T5 smoke gate: a stationarized smoke config so the exog_stationarize END-TO-END
# path (diff transforms + feature build + NN fit on differenced exog) is smoke-able
# on 1 ticker / 1 seed. run_kind stays "smoke" (no separate result group needed).
_SMOKE_IN = os.path.join(_SRC, "config.smoke.yaml")
_SMOKE_OUT = os.path.join(_SRC, "config.smoke_stationary.yaml")


def _flag_block(extra_line: str) -> str:
    return (
        "\n"
        "  # ── stationarized-exog arm (GENERATED — do not hand-edit) ──────────────────\n"
        f"  {extra_line}\n"
        "  # produced by scripts/gen_run2b_config.py. When exog_stationarize is true,\n"
        "  # main.py overrides every exogenous feature's transform per\n"
        "  # features/stationarize.py EXOG_TRANSFORM_TABLE at load time (the single\n"
        "  # source of truth). Re-run the generator after editing the base config or\n"
        "  # the transform table; never edit run_kind/transforms here.\n"
        "  exog_stationarize: true\n"
    )


# An actual YAML key `exog_stationarize:` at the start of a line (any indent),
# NOT a substring inside a comment or description. Anchored so a future doc/comment
# mentioning the flag in the base config doesn't wrongly trip the guard.
_FLAG_KEY_RE = re.compile(r"(?m)^[ \t]*exog_stationarize[ \t]*:")


def inject_stationarize(src_text: str, src_run_kind: str, dst_run_kind: str) -> str:
    """Return ``src_text`` with the ``run_kind`` value rewritten to ``dst_run_kind``
    and ``exog_stationarize: true`` injected right after it inside ``experiment:``.

    Text copy (not a YAML round-trip) so every comment survives and re-running picks
    up base-config edits with zero drift. Raises if the base already declares the
    flag (must live only in the generated config) or if the run_kind anchor is gone.
    """
    if _FLAG_KEY_RE.search(src_text):
        raise ValueError(
            "base config already declares an 'exog_stationarize:' key — the generator "
            "expects the base config to NOT carry the flag. Remove it; the flag belongs "
            "only in the generated stationarized config."
        )
    anchor = f"  run_kind: {src_run_kind}\n"
    if anchor not in src_text:
        raise ValueError(
            f"anchor {anchor!r} not found in the base config; the experiment block "
            "structure changed. Check the run_kind value passed to inject_stationarize."
        )
    if src_run_kind == dst_run_kind:
        note = f"# This file is the base config plus the flag below (run_kind stays {dst_run_kind}),"
    else:
        note = f"# This file is the base config with run_kind set to {dst_run_kind} and the flag below,"
    replacement = f"  run_kind: {dst_run_kind}\n" + _flag_block(note)
    # replace(count=1) hits the first (and only) run_kind line; a stray earlier match
    # would land the flag in the wrong block, which the round-trip equality test
    # (test_run2b_config_is_base_config_*) catches by failing to find the flag.
    return src_text.replace(anchor, replacement, 1)


def generate(src_text: str) -> str:
    """Back-compat wrapper: config.yaml (run2) -> run2b stationarized arm."""
    return inject_stationarize(src_text, "run2", "run2b")


def _write(in_path: str, out_path: str, src_kind: str, dst_kind: str) -> None:
    with open(in_path) as fh:
        out_text = inject_stationarize(fh.read(), src_kind, dst_kind)
    with open(out_path, "w") as fh:
        fh.write(out_text)
    print(f"wrote {os.path.relpath(out_path)} ({out_text.count(chr(10))} lines)")


def main() -> int:
    _write(_IN, _OUT, "run2", "run2b")            # run #2b stationarized arm
    _write(_SMOKE_IN, _SMOKE_OUT, "smoke", "smoke")  # T5 stationarized smoke gate
    return 0


if __name__ == "__main__":
    sys.exit(main())
