"""Check Geodude 4.0 as installed from its wheel, not from a checkout (#208).

The 3.x application, last released as ``v3.1.0``, also imported as ``geodude``. Run
with the clean environment's interpreter, in isolated mode (``python -I``), from outside the
repository, after the installed ``geodude-recycling --seed 0 --record DIR``::

    python -I check_installed.py <repository> <DIR>/recycling.json

It fails unless:
- ``geodude`` is the installed 4.x wheel, in the environment's site-packages, not an
  editable or path install, and no path on ``sys.path`` is inside the repository;
- no module that only the 3.x application had (``LEGACY``, less any the 4.x package
  has itself) is importable, and the wheel ships none;
- the installed command's outcome decodes strictly as a ``RecyclingRun``, succeeded,
  and names this install, by its version and no VCS commit, as its geodude source.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path

LEGACY = frozenset(
    {
        "__main__",
        "bt",
        "chat",
        "cli",
        "config",
        "console",
        "demo_loader",
        "demos",
        "grasp_source",
        "panels",
        "primitives",
        "vention_base",
    }
)
"""The modules and subpackages of ``geodude`` that only the 3.x application had, at
``v3.1.0``, the last 3.x release, removed at the cutover (#212)."""


def _modules(package: Path) -> set[str]:
    """The top-level modules and subpackages of the package directory ``package``."""
    return {
        p.stem
        for p in package.iterdir()
        if (p.suffix == ".py" and p.stem != "__init__") or (p / "__init__.py").is_file()
    }


def main(repository: Path, report_path: Path) -> list[str]:
    failures: list[str] = []
    repository = repository.resolve()
    import geodude

    installed = Path(geodude.__file__).resolve().parent
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    if installed.parent != site:
        failures.append(f"geodude loads from {installed}, not {site}")
    inside = [p for p in sys.path if p and Path(p).resolve().is_relative_to(repository)]
    if inside:
        failures.append(f"sys.path reaches into the repository: {inside}")

    dist = importlib.metadata.distribution("geodude")
    if not dist.version.startswith("4."):
        failures.append(f"geodude {dist.version} installed, not 4.x")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("dir_info") or not direct.get("url", ".whl").endswith(".whl"):
        failures.append(f"geodude installed from {direct}, not a wheel")

    legacy = LEGACY - _modules(installed)  # a 4.x module may reuse a 3.x name
    importable = sorted(m for m in legacy if importlib.util.find_spec(f"geodude.{m}"))
    if importable:
        failures.append(f"legacy modules importable: {importable}")
    shipped = {
        Path(str(f)).parts[1].removesuffix(".py")
        for f in dist.files or ()
        if str(f).startswith("geodude/")
    }
    if shipped & legacy:
        failures.append(f"the wheel ships legacy modules: {sorted(shipped & legacy)}")

    from ssrobot import loads

    from geodude.outcomes import RecyclingRun, Source

    run = loads(report_path.read_bytes(), RecyclingRun)
    checks = [a.checks for a in run.attempts]
    if not (run.cleared and all(a.succeeded for a in run.attempts)):
        failures.append(f"the installed run failed: {checks}")
    named = next(s for s in run.sources if s.name == "geodude")
    if named != Source(name="geodude", version=dist.version, commit=None):
        failures.append(f"the outcome names {named}, not this install")

    print(f"geodude {dist.version} from {installed}")
    print(f"legacy-only modules, none importable: {', '.join(sorted(legacy))}")
    print(f"installed run: {json.dumps(checks)}")
    return failures


if __name__ == "__main__":
    failures = main(Path(sys.argv[1]), Path(sys.argv[2]))
    for failure in failures:
        print(f"FAIL: {failure}")
    sys.exit(1 if failures else 0)
