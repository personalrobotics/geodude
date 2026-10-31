# Frozen cutover corpus evaluation

The one run of the Geodude 4.0 cutover corpus (seeds 2000-2039), declared on #213 before
it ran, with its thresholds: at least 180 of 200 objects recycled and at least 30 of 40
rounds cleared. The result is on #213: 199 of 200 objects recycled and 39 of 40 rounds
cleared, valid, with no errored round and none rerun alone. The corpus is now spent, and
`scripts/evaluate_recycling.py` refuses it.

- `20261031-070726-cutover.json.gz`: the evaluation, exactly as the evaluator wrote it,
  compressed with `gzip -n -9 -c 20261031-070726-cutover.json > 20261031-070726-cutover.json.gz`.
- `20261031-070726-cutover.log`: the run's log: the commit and whether the tree was
  dirty, the command, the host's load and CPU count, the installed versions, and the
  evaluator's summary and exit status.

## How it ran

- Source: personalrobotics/geodude@3bddbef9933ed12802a00d80b6de9e6f0e102086, a fresh
  worktree, not dirty, with `uv sync --locked`.
- Command, from `v4/`, with the default job count (16):

  ```sh
  OPENBLAS_CORETYPE=Haswell uv run --locked python scripts/evaluate_recycling.py cutover runs/20261031-070726-cutover.json
  ```

- Host: Linux-7.0.0-38-generic-x86_64 (glibc 2.39), Python 3.13.3, AMD Ryzen
  Threadripper PRO 7975WX, 64 CPUs, OpenBLAS kernel Haswell. The evaluation records
  the dependency sources and the host itself (`sources`, `host`, `load`, `load_after`).

## Verify it

| File | SHA-256 |
|---|---|
| `20261031-070726-cutover.json` (the original) | `571b9c9d503898d390d5caa47dbe6f6afa9b404b0dd0836fabfba32406cc6abf` |
| `20261031-070726-cutover.json.gz` | `516d887915435f98781978fc333b9732d91dd785b32fbe681470138b735864a7` |

```sh
gzip -dc 20261031-070726-cutover.json.gz | shasum -a 256
```

prints the original's SHA-256.
