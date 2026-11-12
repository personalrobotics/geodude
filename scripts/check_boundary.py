"""Fail unless importing Geodude 4.0 loads none of the legacy stack (#208).

Geodude 4.0 is built on ssrobot and ssgrasping only. The legacy app's mj_manipulator,
mj_environment, and asset_manager, and the old mj-viser distribution, must never load. The
new viewer, ssmjviser, is also imported as ``mj_viser``, so the old one is told apart by
its distribution name.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import sys

PROHIBITED = ("mj_manipulator", "mj_environment", "asset_manager")


def main() -> int:
    import geodude

    for module in pkgutil.walk_packages(geodude.__path__, "geodude."):
        importlib.import_module(module.name)
    loaded = sorted(n for n in sys.modules if n.split(".")[0] in PROHIBITED)
    distributions = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    old_viewer = "mj-viser" in distributions or "mj_viser" in distributions
    if loaded or old_viewer:
        print(f"FAIL: geodude loads {loaded}; old mj-viser installed: {old_viewer}")
        return 1
    print(f"OK: geodude from {geodude.__file__} loads none of {', '.join(PROHIBITED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
