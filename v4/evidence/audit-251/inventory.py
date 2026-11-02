"""The quantitative inventory for geodude#251: reproducible counts of ssrobot,
ssgrasping, and Geodude (legacy and ``v4/`` apart) at fixed commits.

Everything is read from git objects at the named commits, never from a working tree, so
a dirty checkout cannot change the result. Run from anywhere::

    python inventory.py --ssrobot PATH --ssgrasping PATH --geodude PATH > inventory.json

Each ``PATH`` is a clone holding the commits in ``COMMITS``. Python 3.11 or later; the
standard library only.

What is counted, per repository part and category:
- ``lines``: physical lines of every text file; ``code``: lines that are neither blank
  nor only a comment (``#``, Python and shell files only).
- ``files``.
- For Python files: modules, classes, functions (including methods), the top-level
  package's ``__all__`` exports, the largest modules and functions in production code,
  and every import that crosses a repository boundary.

Categories, by path (first match wins, see ``category``):
- ``production``: a package's source tree (``src/``);
- ``tests``: ``tests/``;
- ``docs``: Markdown anywhere, and ``docs/``;
- ``scripts``: ``scripts/``, ``tasks/``;
- ``generated``: ssrobot's ``schemas/`` and ``conformance/``, written by its own tools;
- ``evidence``: committed result artifacts (``evidence/``);
- ``examples``: example packages and reference fixtures (``examples/``, ``references/``);
- ``config``: everything else that is text (``pyproject.toml``, CI, ignore files).

Excluded: lockfiles (``uv.lock``), binary files (any NUL byte), and this audit's own
directory. Nothing is vendored, built, or downloaded into these trees at these commits.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any

COMMITS = {
    "ssrobot": {"baseline": "116f3c8", "snapshot": "0ecbd8d"},
    "ssgrasping": {"baseline": "591f234", "snapshot": "d29a4b4"},
    "geodude": {"baseline": "05f6359", "snapshot": "0618279"},
}
"""The audited snapshots, after ssgrasping#34, ssgrasping#28's estimates and geodude#248,
and the baselines: each repository's last main commit before Geodude 4.0's arc began
(geodude 5efac0c, 2026-10-20, which started ``v4/``)."""

PACKAGES = {"ssrobot": "ssrobot", "ssgrasping": "ssgrasping", "geodude": "geodude"}
OWN = "v4/evidence/audit-251/"
LARGEST = 15


def part(repo: str, path: str) -> str:
    """Which part of a repository ``path`` belongs to: Geodude's legacy application and
    ``v4/`` are counted apart."""
    if repo != "geodude":
        return repo
    return "geodude-v4" if path.startswith("v4/") else "geodude-legacy"


def category(repo: str, path: str) -> str | None:
    """``path``'s category, or None when it is excluded."""
    p = PurePosixPath(path.removeprefix("v4/") if repo == "geodude" else path)
    top = p.parts[0] if len(p.parts) > 1 else ""
    if path.startswith(OWN) or p.name == "uv.lock" or p.suffix == ".lock":
        return None
    if top == "src":
        return "production" if p.suffix == ".py" else "production_data"
    if top == "tests":
        return "tests"
    if p.suffix == ".md" or top == "docs":
        return "docs"
    if top in ("scripts", "tasks"):
        return "scripts"
    if repo == "ssrobot" and top in ("schemas", "conformance"):
        return "generated"
    if top == "evidence":
        return "evidence"
    if top in ("examples", "references"):
        return "examples"
    return "config"


def git(repo_path: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", repo_path, *args], check=True, capture_output=True).stdout


def files(repo_path: str, commit: str) -> dict[str, bytes]:
    """Every blob at ``commit``, by path."""
    listing = git(repo_path, "ls-tree", "-r", "-z", commit).split(b"\0")
    entries = [e.split(b"\t", 1) for e in listing if e]
    blobs = [(meta.split()[2].decode(), name.decode()) for meta, name in entries]
    request = "".join(f"{sha}\n" for sha, _ in blobs).encode()
    out = subprocess.run(
        ["git", "-C", repo_path, "cat-file", "--batch"],
        input=request,
        check=True,
        capture_output=True,
    ).stdout
    contents: dict[str, bytes] = {}
    at = 0
    for _, name in blobs:
        header_end = out.index(b"\n", at)
        size = int(out[at:header_end].split()[2])
        contents[name] = out[header_end + 1 : header_end + 1 + size]
        at = header_end + 1 + size + 1
    return contents


def code_lines(path: str, text: str) -> int:
    if not path.endswith((".py", ".sh")):
        return sum(1 for line in text.splitlines() if line.strip())
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def python(path: str, text: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            imports.append((n.module, [a.name for a in n.names]))
        elif isinstance(n, ast.Import):
            imports.extend((a.name, []) for a in n.names)
    exports: list[str] | None = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets
        ):
            exports = [e.value for e in getattr(n.value, "elts", []) if isinstance(e, ast.Constant)]
    return {
        "classes": len(classes),
        "functions": len(functions),
        "function_sizes": [
            (f"{path}:{f.name}", (f.end_lineno or f.lineno) - f.lineno + 1) for f in functions
        ],
        "imports": imports,
        "exports": exports,
    }


def measure(repo: str, repo_path: str, commit: str) -> dict[str, Any]:
    resolved = git(repo_path, "rev-parse", commit).decode().strip()
    totals: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    modules: dict[str, list[tuple[str, int]]] = defaultdict(list)
    functions: dict[str, list[tuple[str, int]]] = defaultdict(list)
    crossing: dict[str, Counter[str]] = defaultdict(Counter)
    exports: dict[str, list[str]] = {}
    excluded: Counter[str] = Counter()
    for path, blob in sorted(files(repo_path, resolved).items()):
        kind = category(repo, path)
        if kind is None or b"\0" in blob:
            excluded["lockfile_or_audit" if kind is None else "binary"] += 1
            continue
        text = blob.decode("utf-8", errors="replace")
        where = part(repo, path)
        counts = totals[where][kind]
        counts["files"] += 1
        counts["lines"] += len(text.splitlines())
        counts["code"] += code_lines(path, text)
        if not path.endswith(".py"):
            continue
        parsed = python(path, text)
        if parsed is None:
            continue
        counts["modules"] += 1
        counts["classes"] += parsed["classes"]
        counts["functions"] += parsed["functions"]
        if kind == "production":
            modules[where].append((path, code_lines(path, text)))
            functions[where].extend(parsed["function_sizes"])
            name = PurePosixPath(path).name
            if name == "__init__.py" and PurePosixPath(path).parent.name == PACKAGES[repo]:
                exports[where] = sorted(parsed["exports"] or [])
        for module, names in parsed["imports"]:
            root = module.split(".")[0]
            if root in PACKAGES.values() and root != PACKAGES[repo]:
                key = f"{kind}:{root}"
                crossing[where][key] += 1
                for n in names:
                    crossing[where][f"{key}.{n}"] += 0  # the names imported, listed once
    out: dict[str, Any] = {"commit": resolved, "excluded": dict(excluded), "parts": {}}
    for where, by_kind in sorted(totals.items()):
        imported = crossing[where]
        out["parts"][where] = {
            "categories": {k: dict(sorted(c.items())) for k, c in sorted(by_kind.items())},
            "public_exports": exports.get(where),
            "largest_modules": sorted(modules[where], key=lambda m: -m[1])[:LARGEST],
            "largest_functions": sorted(functions[where], key=lambda f: -f[1])[:LARGEST],
            "cross_repository_imports": {
                k: v for k, v in sorted(imported.items()) if "." not in k.split(":", 1)[1]
            },
            "cross_repository_names": sorted(
                k.split(":", 1)[1] for k in imported if "." in k.split(":", 1)[1]
            ),
        }
    return out


def schemas(repo_path: str, commit: str) -> list[str]:
    listing = git(repo_path, "ls-tree", "--name-only", commit, "schemas/").decode().split()
    return sorted(PurePosixPath(p).name.removesuffix(".json") for p in listing)


def scripts(repo_path: str, commit: str, prefix: str) -> dict[str, str]:
    """The installed commands a pyproject declares."""
    import tomllib

    try:
        text = git(repo_path, "show", f"{commit}:{prefix}pyproject.toml").decode()
    except subprocess.CalledProcessError:
        return {}
    return dict(tomllib.loads(text).get("project", {}).get("scripts", {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for repo in COMMITS:
        parser.add_argument(f"--{repo}", required=True, help=f"a clone of {repo}")
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"commits": COMMITS, "python": sys.version.split()[0]}
    for repo, commits in COMMITS.items():
        path = getattr(args, repo)
        report[repo] = {when: measure(repo, path, c) for when, c in commits.items()}
        if repo == "ssrobot":
            report[repo]["schemas"] = {when: schemas(path, c) for when, c in commits.items()}
        prefixes = ("", "v4/") if repo == "geodude" else ("",)
        report[repo]["commands"] = {
            when: {p or "root": scripts(path, c, p) for p in prefixes}
            for when, c in commits.items()
        }
    json.dump(report, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
