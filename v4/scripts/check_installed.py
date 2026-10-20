"""Check Geodude 4.0 as installed from its wheel, not from a checkout (#208).

Both Geodude 4.0 and the legacy app import as ``geodude``. Run with the clean
environment's interpreter, in isolated mode (``python -I``), from outside the
repository, after the installed ``geodude-recycling --seed 0 --record DIR``::

    python -I check_installed.py <repository> <DIR>/recycling.json

It fails unless:
- ``geodude`` is the installed 4.x wheel, in the environment's site-packages, not an
  editable or path install, and no path on ``sys.path`` is inside the repository;
- no module that only the legacy app has (``<repository>/src/geodude``) is importable,
  and the wheel ships none;
- the installed command's report succeeded and names this install as its source.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import sys
import sysconfig
from pathlib import Path


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

    legacy = _modules(repository / "src" / "geodude") - _modules(installed)
    if not legacy:
        failures.append("found no legacy-only modules to check")
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

    report = json.loads(report_path.read_text())
    if not report["succeeded"]:
        failures.append(f"the installed run failed: {report['checks']}")
    if report["sources"]["geodude"] != {"version": dist.version, "source": direct or None}:
        failures.append(f"the report names {report['sources']['geodude']}, not this install")

    print(f"geodude {dist.version} from {installed}")
    print(f"legacy-only modules, none importable: {', '.join(sorted(legacy))}")
    print(f"installed run: {json.dumps(report['checks'])}")
    return failures


if __name__ == "__main__":
    failures = main(Path(sys.argv[1]), Path(sys.argv[2]))
    for failure in failures:
        print(f"FAIL: {failure}")
    sys.exit(1 if failures else 0)
