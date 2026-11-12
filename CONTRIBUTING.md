# Contributing

This repository is part of the [robot-code workspace](https://github.com/personalrobotics/robot-code).
All projects in the workspace share a common layout (`src/` + `tests/`), a single
package manager (`uv`), and a common CI setup.

## Development setup

```bash
git clone https://github.com/personalrobotics/robot-code
cd robot-code
./setup.sh
```

This clones every sibling repo, including this one, and runs `uv sync` for the shared
workspace. Geodude is not a member of it: its dependencies (ssrobot, ssgrasping, and
ssmjviser, which imports as `mj_viser`) cannot share an environment with the
workspace's 3.x stack. So it has its own pinned environment, `uv sync --locked` in this
directory.

## Running tests and linters

From inside this repo:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

From the workspace root, you can also run the cross-repo integration suite:

```bash
uv run pytest tests/integration -v
```

## Pull requests

- Branch from `main`. Open a PR with a clear summary and test plan.
- Per-repo CI (ruff + pytest) must pass.
- Cross-repo integration CI in robot-code runs automatically when this repo's
  `main` advances, and on scheduled nightly runs.
- Review is by [@siddhss5](https://github.com/siddhss5) (enforced via CODEOWNERS).

## Releases

Geodude releases are source releases on GitHub. The committed lock is part of the
release: it fixes the ssrobot, ssgrasping, robot, and object packages that were checked
together. PyPI publication is not part of the 4.0.0 release.

1. In a release pull request, set the project version, move the changelog entry out of
   `Unreleased`, and update the lock with `uv lock`.
2. Merge only after CI has built the wheel, installed it outside the checkout, run the
   installed command, and checked its outcome.
3. Wait for the same `test` job to pass on the merge commit. Tag that exact commit as
   `vX.Y.Z`; do not tag a pull-request head.
4. Create the GitHub release from that tag, using the matching changelog section as its
   notes.
5. In a clean clone of the tag, run `uv sync --locked` and the README's one-can command.
   Record the commit and result in the release notes.

## Package manager: uv only

We use [uv](https://docs.astral.sh/uv/) exclusively. **Do not use pip.**
The workspace layout in robot-code relies on uv's workspace resolution.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
