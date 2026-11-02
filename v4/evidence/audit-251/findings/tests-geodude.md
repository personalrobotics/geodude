# Test audit: Geodude 4.0 (`v4/tests`, installed-command and evaluator gates)

Snapshot: geodude `0618279` (a worktree at the snapshot). Read-only audit for
personalrobotics/geodude#251, "Test audit". Policy: the engineering instructions (`AGENTS.md`), Testing policy.
Paths below are relative to `v4/` unless they start with `.github/`.

## Cost evidence

Per-test wall time from the timestamps in CI run 41338900721 (commit `c4ca3b2`, the
`v4` job, GitHub `ubuntu-latest`, one pytest process). That run predates
`test_registered_control.py`; its cost is an **estimate**, not a measurement (the run on
`0618279`, 41369850922, was still in progress during this audit).

| Test | Rounds run | Wall time |
|---|---|---|
| `test_four_cans` | 1 x four-can `recycle` (seed 4) | 28 s |
| `test_object_types` | 9 x one-object `sort_all` (seed 0) | 57 s |
| `test_planning_latency` | 2 x the same four-can round (one with 0-3 s planner delays) | 98 s |
| `test_recovery` | none (constructed `SkillResult`s) | <0.01 s |
| `test_recycling` (one can) | 5 x one-can `demo` (seed 0 x3, seed 0 worktop, seed 3) | 24 s |
| `test_recycling` (CLI refusal) | none | <0.01 s |
| `test_regions` | none (563 rotations) | 0.04 s |
| `test_sort_all` | 2 x five-object `sort_all` (seed 0) via `main --record` | 74 s |
| `test_registered_control` x2 | 3 x three-object `sort_all` (seed 0) + 2 x one-can retry (seed 1) | **est. 60-90 s** (uncertain) |
| CI "Check the installed wheel" | 1 x one-can `demo` (seed 0) from the wheel, plus checks | 10 s |

Pytest total on `c4ca3b2`: 281.5 s. With `test_registered_control`: est. ~6-6.5 min.
Roughly **75-80 s** of that is runs whose only purpose is to repeat a round to check
determinism (see "Shared generation" below).

## Per-test table

Legend for "CD" (change-detector) and "Re" (reimplements production logic): no / partial /
yes. "Overlap" names other tests or the development/cutover evaluator corpus.

### `tests/test_recycling.py`

| Scenario / assertion group | Protects (external behavior, invariant, boundary, failure mode) | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| One-can bin round: `checks` all True, `succeeded`, nothing attached (`:107-108`) | The single-can `demo()` path (`recycling.py:753`) actually recycles: grasp, drop, landing confirmed, can inside the bin's prl_assets interior. `demo()` is a separate code path from `recycle`/`sort_all` (fixed `arm_for`, no race), and it is the path the installed command runs. | The evaluator and every other MuJoCo test go through `recycle` (`recycling.py:941`), not `demo`. Only CI's installed run (`ci.yml:114`, `check_installed.py:71`) also exercises `demo` to the same oracle. | no | no | Unique for `demo`. |
| Left arm, seed 3 (`:78`, `:111-114`) | Arm chosen by the can's side where the other arm cannot reach: step 3 of #201. | `test_four_cans`/`sort_all` race both arms, but via `recycle`'s `Candidates`, not `arm_for`. | partial (bin name `recycle_bin_1`) | no | Unique for `demo`'s `arm_for`. |
| Worktop placement, step 1 (`:77`, `:115-116`) | `place_on_worktop` + `worktop_outcome`: upright, base on worktop, at target within tolerance. | **None.** No other test, and no corpus, places on the worktop. | no | no | Only coverage of `--to worktop` / `place`. |
| Repeat identical incl. trace sha256 (`:76`, `:87`, `:117`) | Same seed gives same outcome and byte-identical trace on the `demo` path. | Determinism is also asserted by `test_sort_all`, `test_planning_latency` (stronger: trace digests under perturbed planning time), `test_registered_control` (twice), and here a third time by `recorded_matches_repeat` (`:88`). | no | no | `second` run (~5 s) is redundant with `recorded_matches_repeat` for the outcome; it is the only one comparing the trace hash on `demo`. |
| `recorded_matches_repeat` (`:88`) | The CLI (`main --record`) wires to `demo` and reports what `demo` returns. | CI installed run checks `succeeded` and sources only. | no | no | Cheap given `main` is run anyway. |
| Scene manifest: prl_assets source, files == staged files (`:100-102`) | Staging provenance: the manifest names every staged file and its source. | `test_sort_all` compares `scene_identity` across runs (`test_sort_all.py:73-74`), not file completeness. | no | no | Unique, cheap (no extra round). |
| Object / fixture name lists (`:103-104`), `bin.name == recycle_bin_0`, `arm == right` (`:105-106`) | Mostly naming convention of `stage_scene`. `arm == right` for seed 0 is the side rule's positive case. | `arm` side rule overlaps left-seed case. | **yes** for names (freezes `can_0`, `recycle_bin_0/1` naming with no semantic oracle) | no | Low value; zero cost. |
| `drop` stages `== ["preflight","transfer","release"]` (`:109`) | Freezes ssgrasping's drop stage decomposition. The semantic content ("release confirmed") is already in `checks["released"]`. | `checks` at `:107`. | **yes** | no | Change-detector of another repo's internal stage list. |
| `landing == [["observed", True]]` (`:110`) | Freezes the landing skill's stage list. Semantic content is `checks["landed"]` (scored `confirmed`). | `checks` at `:107`. | **yes** | no | Change-detector. |
| `version.startswith("4.")`, `seed` (`:98-99`) | Report provenance. | `check_installed.py:50-51`, `:73-74` check the same against the real install. | partial | no | Redundant with the installed gate. |
| Bundle: snapshots >= 2, world materializes and matches manifest (`:83-95`, `:118`) | Artifact boundary: a Geodude run's bundle verifies and its world materializes without a runtime. | `test_sort_all` opens bundles (`:62`) but does not materialize the world. ssrobot's own artifact tests likely cover `materialize_world` generically (out of scope here; unverified). | partial (`>= 2` snapshot count) | no | Unique for Geodude's real scene. |
| `test_the_demo_refuses_impossible_can_counts_before_starting` (`:121-130`) | CLI trust boundary: impossible `--cans` refused with exit 2 before any setup (would otherwise report success on an empty scene). | None. | no | no | Keep. `--mix` refusals (`recycling.py:1257-1258`) are not covered (gap noted, no recommendation). |

Doc inaccuracy: the module docstring (`:23-24`) says CI "also runs the same command from the
built wheel"; CI runs `geodude-recycling --seed 0 --record` (`ci.yml:114`), not this test.

### `tests/test_four_cans.py`

| Assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| Run succeeds, 4 cans (`:85-86`) | Multi-can `recycle` clears four cans. | Development/cutover corpora run `sort_all` -> `recycle` (`recycling.py:1170`) on 200 objects; `test_planning_latency` runs **this exact round** twice and asserts `code == 0` (`test_planning_latency.py:82`). | no | no | Outcome fully duplicated by `test_planning_latency`'s prompt run. |
| Race: both arms raced, selected == arm, preferred arm wins when it succeeds (`:88-93`) | Race selection is deterministic by priority, not by planning speed. | `test_planning_latency` proves selection is unchanged under delays (outcome equality), not the priority rule itself. | no | partial (restates the `PREFERRED` rule) | Keep the priority oracle. |
| Prediction timeline: first can on snapshot, each later can raced on exactly one `PREDICTED` record made during the transfer, `CONTINUED` decisions after it ended, pregrasp decision matches (`:102-111`) | The core concurrency claim of #223: plan the next can on the prediction while the current one is carried, decide continuation only after the transfer ends. A genuine causal invariant from the trace. | Exercised (not asserted) by every corpus round and by `test_planning_latency` (digest equality only). | no | no | **Unique.** |
| `approach.lineage == [base]` (`:95`) | Prediction lineage is a single base. | none | partial | no | |
| `approach["kept"] == "the feasibility check, planned before"` (`:98`) | That the race's approach plan is kept, not replanned. | none | **yes**: exact prose string literal defined in **ssgrasping** (`ssgrasping/src/ssgrasping/_skill.py:364`); a rewording there fails Geodude CI with no behavior change | no | Change-detector across repos. The behavior (no replan) has no structured field here; the oracle is prose. |
| `[e.stage for e in lookahead.drop] == ["release"]` (`:99`) | "Drops do not retreat" (docstring `:9`). | none | partial (exact stage list; the design claim is "no retreat lookahead") | no | |

### `tests/test_planning_latency.py`

| Assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| Delayed run's outcomes, end time, and every trace-kind count + digest equal the prompt run's (`:81`) | Host-independence: wall-clock planning latency never changes simulated behavior (the clock is not stepped while skills wait only on plans, #242). A real, otherwise unobservable invariant: the corpus would show it only as cross-host irreproducibility. | Strictly subsumes plain repeat-determinism of this round (and largely of `test_recycling`'s repeat on a different path). | no (equality, not frozen values) | no | **Unique; strongest determinism gate in the suite.** |
| `max(delayed) == 3.0` (`:80`) | The perturbation actually reached its longest delay. | none | no | no | Sanity of the perturbation, fine. |
| `prompt.code == 0` (`:82`) | Round succeeds. | `test_four_cans:85` (same round). | no | no | |
| Coupling | Monkeypatches private `ssrobot.planning._cbirrt.CBiRRT.plan` and imports `ssrobot._wire.fingerprint` (`:31-32`, `:65-74`). | | | | Brittle to ssrobot internals; not a deletion reason. |

### `tests/test_sort_all.py`

| Assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| `cleared`, 5 objects, none attached (`:69-71`) | The `--mix` CLI path clears a mixed five-object table (#201 step 5), the legacy demo's default round. | Development corpus seed 0 is **the identical round** (`evaluate_recycling.py:113-126`: `mix(0, 5)`, seed 0, `sort_all`) but the evaluator is not in CI. `test_registered_control` A runs `sort_all` on `mix(0, 3)`, seed 0 (different scene). | no | no | Only CI gate on a full five-object mixed round. |
| Bundle manifests name same world / scene / model (`:73-74`) | Recording identity is reproducible. | `test_recycling` bundle checks (one run). | no | no | |
| Second run's deterministic outcome byte-identical (`:58-60`, `:75-76`) | Determinism of `sort_all` on mixed kinds, including final poses. | `test_planning_latency` (cans, `recycle`, stronger); `test_registered_control` (B twice, `mix(0,3)`, control record but not final poses). | no | partial: `_outcome` (`:39-53`) hand-selects "the part the seed alone decides" | The second run costs ~37 s for a determinism-only check. |

### `tests/test_object_types.py`

| Assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| Each of the 9 `MIX_KINDS`, staged alone, succeeds at first attempt and rests in a bin (`:55-60`) | Per-type grasp regions and drop zones work for every legacy type (#201 step 4); a broken region for one type is diluted in mixed rounds. | Corpus `summary.by_kind` covers all kinds statistically (cutover: 199/200), but not in CI and allowing retries/recovery. `test_sort_all` covers only seed 0's draw. | no | no | **Keep.** Semantic oracle only (`succeeded` and `bin`). |

### `tests/test_recovery.py`

| Assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| `safe_release(_failure(result))` decision table over 3 failure kinds x 3 detail strings, negatives for wrong reason, not attached, held not False, and `None` (`:79-98`) | Safety boundary: fingers open only where known safe; never decided from prose `detail` (the #252 bug class). Detach-lost only when the skill reported not held (#253). | Corpus exercises recovery rarely (cutover: 2 of 200 recycled after recovery, 0 after retry). An end-to-end run cannot vary `detail` or force the negative cases economically. | no | partial: the expected-action table restates `safe_release`'s three branches (`recycling.py:849-864`), but it is the stated safety contract and the detail-invariance and negatives are independent | Isolated test justified by the policy's enumerated-failure-mode clause. **Keep.** Written together with the fix (#252, #253), but each case names a failure ssgrasping's e2e evidence produces (docstring `:10-14`). |

### `tests/test_regions.py`

| Assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| `matrix(pose(T)) == T` for 500 random rotations plus all half turns about {-1,0,1}^3 axes (`:27-41`) | Mathematical invariant: matrix to quaternion round-trip, incl. half turns that silently flipped half of every cylinder's side grasps (#232; docstring `:4-7` says no e2e run reported it). | `test_object_types` now succeeds on cylinders, but success does not require the full grasp set, which is exactly why the bug was missed. | no | no (oracle is the inverse map) | `regions.pose` now delegates to `mujoco.mju_mat2Quat` (`regions.py:51-60`), so the test mostly guards against a regression to a hand-rolled conversion. Cheap. **Keep.** |

### `tests/test_registered_control.py`

| Scenario / assertion group | Protects | Overlap | CD | Re | Notes |
|---|---|---|---|---|---|
| A (evaluated, stub) vs B (privileged channels withheld, perception stand-in): control records equal (`:243`), registrations equal and non-empty (`:244`), final bins equal (`:254`) | "Evaluation steers nothing" and control runs without privileged channels (#248, memory "no privileged control"). Trust boundary between control and evaluation. | none | no | partial: `_control` (`:131-151`) hand-selects the control fields | **Unique. Keep.** |
| B provenance: zero privileged readings, registered pose == latest registration, no `measured` objects, source labels (`:245-253`) | No object pose enters control except via `recycling.register`. | ssrobot's own context tests may cover "registered pose is its registration" generically (other auditors' area; unverified). | partial (source label strings `pose_stub` / `perception_stand_in` are the oracle) | partial: `_provenance` (`:154-183`) re-derives snapshot/registration consistency from the trace | Keep; the Geodude-specific claim is the zero-privileged-readings count. |
| B run twice byte-identical (`:235-236`, `:242`) | Determinism of the unprivileged path. | A == B control already implies control determinism across two runs under different wrappers; `test_planning_latency` is the strong determinism gate. | no | no | Third `sort_all` round (est. 20-30 s) for a repeat-only check. Loses only byte-identity of provenance counts and registration counts, which follow from control. |
| Moved can: first attempt `missed`, table re-registered, retry on a later registration at the moved pose, released into a bin, 2 registrations, 0 privileged readings (`:333-349`) | Retry-after-miss with re-registration, under no privileged reads: the closed-loop recovery path that the corpus exercises rarely (cutover: 0 recycled after retry). | none | partial: `missed.grasp[-1] == ["obstructed", False]` (`:342`) freezes the stage name; `registrations == 2` is semantic | no | **Unique. Keep.** Reaches into `inner._model/_data` (`:280-283`), private. |
| Moved run repeated identically (`:336-337`, `:340`) | Determinism of the injected-perturbation path. | General determinism gates above. | no | no | Repeat-only run (est. 8-15 s). |

### CI gates (`.github/workflows/ci.yml`, `scripts/check_installed.py`, `scripts/check_boundary.py`)

| Gate | Protects | Overlap | Notes |
|---|---|---|---|
| Reference OpenBLAS kernel (`ci.yml:83-86`) | All byte-identity claims in the tests and corpora are made on one BLAS kernel. | none | Keep; precondition for every determinism assertion above. |
| `check_boundary.py` in the dev env (`ci.yml:91-92`) | Importing every `geodude.*` module loads no legacy stack and no old `mj-viser`. | Same script in the clean wheel env (`ci.yml:113`), which is the user-facing boundary and has no dev dependencies. | Dev-env run is a fail-fast duplicate of the stronger clean-env run (seconds; low priority). |
| Wheel build + clean install + `geodude-recycling --seed 0 --record` + `check_installed.py` (`ci.yml:100-117`) | Release artifact boundary: the 4.x wheel stands alone (site-packages, no repo on `sys.path`, not editable, ships and exposes no legacy-only module, `check_installed.py:41-68`), and the installed command succeeds and names this install (`:70-74`). | `test_recycling` runs the same command in-tree; it cannot prove the wheel boundary. | **Unique. Keep.** `check_installed.py:57-58` fails if it finds no legacy modules to check: a good guard against a vacuous check. |
| `evaluate_recycling.py` own checks (`:475-513`): spent corpus / spent `--round` seed refused, frozen corpus whole, clean tree, reference kernel, never overwrite, atomic write (`_write`, `:399-404`), invalid evaluation exits nonzero (`:407-412`), `--classify` derived artifact with source SHA-256 (`:415-430`) | Evidence integrity of the frozen-corpus claims (a spent corpus is never rerun; artifacts never overwritten or partially written). | **No test and no CI step runs the evaluator at all.** Its `classify`/`verdict`/`_summary` (`:197-274`), which produce the claimed numbers, are verified only by inspecting the artifacts. | Not a deletion candidate. Observation: the corpus gates are enforced only by argparse refusals that nothing exercises; whether that is a gap worth a test is a decision for the consolidation (an end-to-end `--seeds 0` dev run would be the behavior-level form, at ~40 s). |

## (1) Keep: uniquely protects something

1. `test_recycling.py::test_geodude_recycles_one_can_on_ssrobot`, as a scenario: only coverage of `demo()`'s `arm_for` path (`recycling.py:753-783`) and the **only** coverage of worktop placement (`:77`, `:115-116`), plus manifest completeness and bundle materialization on Geodude's scene.
2. `test_recycling.py::test_the_demo_refuses_impossible_can_counts_before_starting`: CLI boundary.
3. `test_four_cans.py`: the prediction-timeline assertions (`:102-111`) are the only assertion of #223's concurrency claim.
4. `test_planning_latency.py`: host-independence of simulated behavior; the strongest determinism gate.
5. `test_object_types.py`: per-type coverage in CI.
6. `test_sort_all.py`: first run only (the five-object mixed round through `main --mix --record`).
7. `test_recovery.py`: safety decision table; justified isolated test.
8. `test_regions.py`: rotation round-trip invariant; cheap.
9. `test_registered_control.py`: A vs B (trust boundary) and the moved-can retry scenario.
10. CI: BLAS-kernel gate, clean-wheel `check_boundary` + installed run + `check_installed.py`.

## (2) Deletion / merge candidates (with evidence)

None of these deletes a whole test file. Each removes a repeat run or an assertion.

| # | Candidate | Real failure mode considered | Replacing coverage | Unique boundary lost | Saving | Confidence |
|---|---|---|---|---|---|---|
| D1 | `test_sort_all`: drop the second run and the byte-identical comparison (`:58-60`, `:72-76`) | Nondeterminism in `sort_all` (thread scheduling in races, unseeded randomness, contact physics) | `test_planning_latency` (trace digests equal under perturbed planning time on `recycle`, the function `sort_all` calls); `test_registered_control` A == B and B x2 on a mixed `sort_all` | Bit-identical **final poses** of non-can kinds through `main --mix`; `retries=1` path determinism. Nondeterministic contact physics for boxes/bottles would otherwise surface only in the corpus. | ~37 s | medium |
| D2 | `test_registered_control`: drop the B repeat (`:236`, `:242`) | Unprivileged path nondeterministic | A == B control equality across two independent runs (`:243`); `test_planning_latency` | Byte-identity of provenance/registration counts across two B runs (they follow from identical control) | est. 20-30 s | medium-high |
| D3 | `test_registered_control`: drop the moved-can repeat (`:337`, `:340`) | Injected move or retry nondeterministic | General determinism gates (D1/D2 replacement) | Determinism of the retry/re-registration branch specifically | est. 8-15 s | medium |
| D4 | `test_recycling`: drop the `second` `_run` (`:76`, `:87`, `:117`) | `demo` nondeterministic | `recorded_matches_repeat` (`:88`) already compares `main`'s outcome to `first`'s | Trace-hash identity on the `demo` path (outcome identity remains) | ~5 s | high |
| D5 | `test_recycling` `drop` stage list (`:109`) and `landing` stage list (`:110`) | ssgrasping drops a stage or the landing stops being observed | `checks["released"]` and `checks["landed"]` in `:107` assert the semantic outcomes | Nothing semantic; these freeze ssgrasping's stage names | 0 s (maintenance) | high |
| D6 | `test_recycling` object/fixture name lists (`:103-104`) | `stage_scene` naming changes | `bin.name`/`arm` checks and `checks` | None semantic | 0 s | high |
| D7 | `test_four_cans` `kept == "the feasibility check, planned before"` (`:98`) | Race's approach plan is replanned instead of kept | **None structured.** Replace the prose match with a structured field from ssgrasping if one exists or is added (cross-repo; out of this audit's scope). Do not simply delete. | The "kept, not replanned" behavior | 0 s | high that it is a change-detector; the fix is a structured oracle, not deletion |
| D8 | `test_recycling` `version.startswith("4.")` (`:99`) | Report misnames the installed geodude | `check_installed.py:50-51`, `:73-74` against the real wheel | In-tree run only | 0 s | high |
| D9 | CI dev-env `check_boundary.py` (`ci.yml:91-92`) | Legacy stack loads on import | Same script in the clean wheel env (`ci.yml:113`) | Fail-fast before the 5-minute pytest step; a legacy import via a dev-only dependency (does not reach users) | seconds | low priority |

## (3) Expensive runs to generate once and check by several gates

| # | Shared run | Consumers today | Proposal | Saving |
|---|---|---|---|---|
| S1 | `recycling.main(["--seed","4","--cans","4","--record",...])` | `test_four_cans` (`:72`) and `test_planning_latency`'s prompt run (`:43`, `:63`): **the identical command**, run twice per suite | One module/session fixture records it once; `test_four_cans`'s timeline assertions and the latency comparison both read that recording (or merge the two files into one test of the four-can round). | ~28 s |
| S2 | Determinism repeats in five places: `test_recycling` (`second` + `main`), `test_sort_all` (run 2), `test_registered_control` (B x2, moved x2), `test_planning_latency` (prompt vs delayed) | five tests, ~75-80 s of repeat-only rounds | Make `test_planning_latency` the declared determinism gate (it subsumes plain repeats on `recycle`), and keep at most one plain repeat for a path it does not cover (e.g. mixed kinds, D1). See D1-D4. | ~70 s |
| S3 | Seed-0 five-object `sort_all` | `test_sort_all`; development corpus seed 0 (`evaluate_recycling.py:113-126`, same scene and function) | If a dev-corpus smoke ever enters CI, run seed 0 once and let both the evaluator's artifact and `test_sort_all`'s checks read it. Today the evaluator is not in CI, so no saving yet. | 0 now |
| S4 | Seed-0 one-can `demo` | `test_recycling` (3 runs), CI installed run (`ci.yml:114`) | The installed run must come from the wheel, so it cannot be shared with in-tree pytest; within `test_recycling`, D4 removes one of three. | ~5 s |
| S5 | Seed-0 `sort_all`, three objects | `test_registered_control` A, B, B | D2 removes one. A cannot be shared with `test_sort_all` (different object count, so a different staged scene). | est. 20-30 s |

## Uncertainty and limits

- `test_registered_control` wall time is estimated from object counts, not measured.
- Whether ssrobot's or ssgrasping's suites already cover bundle materialization, registration
  consistency, or drop/landing stage semantics is other auditors' scope; marked "unverified" above.
- I did not run any test. Costs are from one CI run on GitHub's runner.
- "Added after implementation": every v4 test landed in the same PR as its implementation
  (`git log` per file). Each states an independent failure in its docstring (`test_regions`
  the half-turn sign, `test_recovery` prose-based decisions, `test_planning_latency` the
  host-dependent clock), so none is flagged as a mere ratification.
