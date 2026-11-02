"""A recycling outcome is refused, never coerced, when it is not a complete, consistent
``RecyclingRun`` (personalrobotics/geodude#246).

The four-can round's ``recycling.json`` (the session's ``four_cans`` recording) decodes.
Copies of it, each tampered with in one way, must each be refused with its own
diagnostic code:
- ``missing``: an attempt without ``succeeded``;
- ``extra``: an attempt with a field the format does not have;
- ``wrong_type``: an attempt numbered by a string;
- ``inconsistent``: an attempt whose checks all passed, recorded as failed;
- ``partial_landing``: a dropped can, released and evaluated, without its landing: the
  entry an attempt was while its landing was still observed;
- ``other_version``: the outcome claimed in another version of the format.

The evidence is ``tampered/``, kept by this test: every tampered copy, and
``diagnostics.json``, each copy's refusal (code, path, message).

Reproduce with ``uv run pytest tests/test_outcome_boundary.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import ValidationError, loads

from geodude.outcomes import RecyclingRun


def _missing(run: dict[str, Any]) -> None:
    del run["attempts"][0]["succeeded"]


def _extra(run: dict[str, Any]) -> None:
    run["attempts"][0]["note"] = "added by hand"


def _wrong_type(run: dict[str, Any]) -> None:
    run["attempts"][0]["number"] = "1"


def _inconsistent(run: dict[str, Any]) -> None:
    run["attempts"][0]["succeeded"] = False


def _partial_landing(run: dict[str, Any]) -> None:
    attempt = run["attempts"][0]
    attempt["landing"] = None
    attempt["times"]["landed"] = None


def _other_version(run: dict[str, Any]) -> None:
    run["version"] = RecyclingRun.VERSION + 1


TAMPERED: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "missing": (_missing, "missing_field"),
    "extra": (_extra, "unknown_field"),
    "wrong_type": (_wrong_type, "wrong_type"),
    "inconsistent": (_inconsistent, "inconsistent_success"),
    "partial_landing": (_partial_landing, "incomplete_landing"),
    "other_version": (_other_version, "version_mismatch"),
}
"""Each tampering, and the code it must be refused with."""


def test_tampered_outcomes_are_refused_each_with_its_own_code(
    artifacts: Path, four_cans: tuple[int, Path]
) -> None:
    _, out = four_cans
    original = (out / "recycling.json").read_bytes()
    run = loads(original, RecyclingRun)
    landed = run.attempts[0]
    assert landed.landing is not None and landed.checks["released"] and landed.succeeded

    copies = artifacts / "tampered"
    copies.mkdir(exist_ok=True)
    diagnostics: dict[str, Any] = {}
    for name, (tamper, _) in TAMPERED.items():
        data = json.loads(original)
        tamper(data)
        text = json.dumps(data, indent=1) + "\n"
        (copies / f"{name}.json").write_text(text)
        try:
            loads(text, RecyclingRun)
        except ValidationError as refused:
            diagnostics[name] = {
                "code": refused.code,
                "path": refused.path,
                "message": refused.message,
            }
        else:
            diagnostics[name] = None
    (copies / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    codes = {name: (d or {}).get("code") for name, d in diagnostics.items()}
    assert codes == {name: code for name, (_, code) in TAMPERED.items()}, diagnostics
    assert len(set(codes.values())) == len(codes)
