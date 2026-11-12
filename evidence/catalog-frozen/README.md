# Frozen catalog corpus evaluation

This is the one run of the frozen catalog corpus, seeds 4000-4059, declared on #267 before it ran. It has no pass threshold: it measures recycling over every catalog kind that stages on the worktop, 30 kinds, each staged 10 times.

The result is on #267. It is valid, with no errored round and none rerun alone.
- **The 18 supported kinds:** 174 of their 180 instances recycled, 172 at the first attempt and 2 after a retry.
- **The 12 unsupported kinds:** 0 of 120, for the reasons recorded on #266.

The corpus is now spent, and `scripts/evaluate_recycling.py` refuses it.

## Files
- **`20261111-215523-catalog-frozen.json.gz`:** the evaluation, exactly as the evaluator wrote it, compressed with `gzip -n -9 -c 20261111-215523-catalog-frozen.json > 20261111-215523-catalog-frozen.json.gz`.
- **`20261111-215523-catalog-frozen.log`:** the run's log: the command, the commit and whether the tree was dirty, the host's load and memory, the installed versions, and the evaluator's summary and exit status.

## How it ran
- **Source:** personalrobotics/geodude@18b6bbdf6da5d723b9599ec5b4e7e4a5a71c04f8, also tagged `v3.1.0`. It ran from a fresh clone, not dirty, after `uv sync --locked`.
- **Command,** from `v4/`, with 8 jobs:

  ```sh
  OPENBLAS_CORETYPE=Haswell uv run python scripts/evaluate_recycling.py catalog-frozen runs/20261111-215523-catalog-frozen.json --jobs 8
  ```

- **Host:** Linux-7.0.0-38-generic-x86_64 (glibc 2.39), Python 3.13.3, AMD Ryzen Threadripper PRO 7975WX, 64 CPUs, OpenBLAS kernel Haswell.
- **Recorded in the evaluation:** the dependency sources (ssrobot e1761b1, ssgrasping 62863ae, sscbirrt 3.16.0, prl_assets 6b5cc8d, geodude_assets e9538a2) and the host (`sources`, `host`, `load`, `load_after`).

## Verify it

| File | SHA-256 |
|---|---|
| `20261111-215523-catalog-frozen.json` (the original) | `537225d758aa782ed71b2b3dabc5069414d97ef68f5cb3e9e1590f29f9182341` |
| `20261111-215523-catalog-frozen.json.gz` | `df637689f53f4913ffe54a211f61285bdd7492f2a6a95a9350ae0b1e8cf4a49a` |

```sh
gzip -dc 20261111-215523-catalog-frozen.json.gz | shasum -a 256
```

prints the original's SHA-256.
