# Geodude 4.0 (v4/): flows 4 and 5, concentration points, #245 and #246

Auditor scope: personalrobotics/geodude#251, Geodude 4.0 only. Snapshot: geodude
`0618279186a833fe9b535fa6c0ef21ebd3604147` (clean), with ssgrasping `d29a4b4` and ssrobot
`0ecbd8d` read only where Geodude calls them. Paths below are relative to `geodude/v4/`
unless prefixed `ssgrasping/` or `ssrobot/`. Nothing was run except read-only `grep`,
`git log`, and a Python read of the committed cutover evidence. Legacy 3.x is not
audited.

Sizes (physical lines): `src/geodude/recycling.py` 1352, `regions.py` 266, `scene.py` 317;
`scripts/evaluate_recycling.py` 520, `calibrate_release.py` 240, `time_startup.py` 235,
`check_installed.py` 86, `check_boundary.py` 35; tests 967.

Responsibility map of `recycling.py` (line ranges, approximate):

| Lines | Responsibility | Size |
|---|---|---|
| 98-192, 809-815 | measured constants and task data (bins, tolerances, mix kinds, sources, BLAS kernel) | ~100 |
| 194-258 | `Arm`, `planners`, `warm_planning` | 65 |
| 261-330 | registration seam (`register`, `pose_stub`, `on_table`, `assume_landed`) | 70 |
| 344-450 | `drive` and the single-object helpers `pick`, `place_on_worktop`, `drop_in_bin` (+`grasp_spec`) | 107 |
| 453-507 | drop/landing specs, `NO_ZONE` | 55 |
| 510-539 | provenance (`sources`, `blas_kernel`) | 30 |
| 542-750 | outcome construction (`_stages`, `_times`, `_transitions`, `_failure`, `worktop_outcome`, `_in_bin`, `bin_outcome`) | 209 |
| 753-783 | `demo` (one-can path) | 31 |
| 786-806 | `candidates` | 21 |
| 817-938 | recovery (`_wait`, `_release`, `safe_release`, `recover`, `_upright_on_worktop`) | 122 |
| 941-1150 | `recycle` (runner, retries, landings, missed re-registration) | 210 |
| 1153-1186 | `sort_all` | 34 |
| 1189-1220 | `_drop_racing_next` | 32 |
| 1233-1352 | `main` (CLI, viewer pacing, bundle and report writing) | 120 |

## Flow 4: recover from a failed skill

Entry points: `geodude-recycling --mix N` → `sort_all` (recycling.py:1313-1316,
1153-1186) → `recycle(retries=RETRIES)` (1170); `--cans N>1` → `recycle(retries=0)`
(1321); `--cans 1` → `demo` (1318), which has **no recovery at all** (753-783).

Path inside `recycle` for one attempt (941-1150):

1. Race: `Candidates(...)` (1025) driven by `run_together` (1026) with `stepped` as tick
   (1005-1013); `selection is None` appends a no-arm entry and continues (1029-1040).
2. Grasp: `drive(ctx, selection.grasp(...), stepped)` (1044) → `run_together`
   (344-352).
3. Drop: `held_drop_spec` (1050, 486-501); `None` means "no zone", later written as the
   Geodude-local failure `NO_ZONE` (474-481, 1119-1121). Otherwise `_drop_racing_next`
   (1078, 1189-1220) drives the drop and starts the next race at the transfer.
4. Outcome entry: a partial dict (1092-1100) appended to `outcomes` immediately; if
   released and `evaluate`, completed later by a `record` closure when the landing ends
   (1101-1116, 1009-1013); otherwise completed now by `bin_outcome` (1118).
5. Missed grasp (`picked.stage.reason is Reason.MISSED`, typed check, 1122-1132): close
   any pending race, re-register the table, maybe requeue, skip recovery.
6. Recovery: only if still attached (1133-1134). `recover(ctx, obj, arm,
   entry["failure"], ...)` (1135-1137) — the input is the **serialized outcome dict**
   produced by `_failure` (606-623) or `NO_ZONE`, not the typed `Stage`.
   `safe_release` (849-864) compares dict strings to `Reason`/`Transition` StrEnums
   (858-862). `recover` (867-927) runs gripper and back-out commands through `_wait`
   (817-824) and returns a third dict shape `{action, released, detail}` stored at
   `entry["recovery"]` (1138).
7. Retry: not released → `break` the whole round (1139-1140); released and under the
   retry budget → re-register the object and requeue if `_upright_on_worktop`
   (1141-1144, 930-938).
8. After the loop: a fifth step loop drains remaining landings (1145-1149).

Concepts a caller/maintainer must hold for this flow: `Arm`, `Candidates`/`Selection`/
race report, `Lookahead`, `SkillExecution`/`SkillResult`/`Stage`/`Reason`/`Evidence`,
`TransitionEstimate`, `landing`, scene revision as "registration", attachment, the
failure dict, the recovery dict, `NO_ZONE`, `RELEASE_IN_PLACE`/`RELEASE_LOST`, `RETRIES`,
queue/attempt counters, `released` (unsettled list), `landings` + `record` closures.

Findings:

- **F4.1 Recovery policy depends on which CLI flag was used** (essential policy, accidental
  variation). One can: no recovery, no retry (753-783). `--cans N`: recovery, no retry
  (`retries=0`, 949, 1321). `--mix`: recovery and one retry (1170, 809-812). Same task,
  three failure policies chosen by staging flag. Nothing documents why `--cans` differs
  from `--mix` beyond history (`--cans` predates #235: `git log` shows recycle in #225,
  `sort_all` in #235).
- **F4.2 Control reads the persisted dict.** `recover`/`safe_release` take the
  `_failure` dict (849-864, 1136), while the missed-grasp branch reads the typed stage
  (1122). `tests/test_recovery.py:85-97` tests `safe_release(_failure(result))`, i.e.
  the decision is coupled to the serializer's keys. Accidental: the decision only needs
  `Stage` (skill, name, reason, cause, held, attached) plus the `NO_ZONE` case.
- **F4.3 Five shapes of one attempt record** produced in `recycle`: no-arm entry
  (1031-1038); partial entry awaiting landing (1092-1099); `bin_outcome` merge (1111 or
  1118); `NO_ZONE` override of `failure` (1121); `recovery` added (1138). Consumers
  guard with `.get` (`scripts/evaluate_recycling.py:172, 188, 212, 283, 297`;
  `tests/test_registered_control.py:318-323`).
- **F4.4 A partial entry can escape.** If the final drain (1145-1149) exhausts
  `MAX_TICKS`, the partial entry (no `succeeded`, `checks`) is returned; `main` then
  fails with `KeyError` at 1322 and the evaluator's `classify` at
  `evaluate_recycling.py:202`. Not observed in evidence; credible, and exactly #246's
  "partial landing entry" criterion.
- **F4.5 Four exhaustion policies for one budget** (`MAX_TICKS`, 169): `drive` cancels
  (349-350); `_drop_racing_next` cancels the drop but leaves the race running and counts
  waits as ticks (1202, 1218-1219) whereas `run_together` counts only steps
  (`ssgrasping/src/ssgrasping/execution.py:194-198`); `_wait` returns a non-terminal
  status silently (817-824), which `_release` reports as "did not open" (836-844); the
  drain stops silently (1145-1149).
- **F4.6 The missed-grasp path (1122-1132) is coherent and essential** (geodude#248;
  `tests/test_registered_control.py:333-349`). Its only accidental cost is being a
  third inline branch of the loop body rather than part of one "after an attempt"
  decision with recovery.

## Flow 5: inspect a run and reproduce its decision

Artifacts and their owners:

| Artifact | Producer | Typed/validated | Holds the "why"? |
|---|---|---|---|
| `bundle/` (manifest, trace, model, scene) | ssrobot `write_bundle` (recycling.py:1333-1335) | strict: `open_bundle` verifies hashes, identities, strict trace decode (`ssrobot/src/ssrobot/bundle.py:121-134`) | motions, predictions (`PREDICTED`), continuation decisions (`CONTINUED`), scene changes with source; **no** skill stages, race selection, recovery or retry decision (`ssrobot/src/ssrobot/trace.py:24-62`) |
| `recycling.json` | `main` (1336-1343) | none; plain `write_text`, not atomic | the only durable record of stages, reasons, race, recovery, transitions |
| evaluation JSON | `evaluate_recycling.py` (433-457, `_write` 399-404 atomic) | `FORMAT` integer only (67); no decoder | per-round `outcomes` (=`sort_all` attempts) + `objects` + verdicts; **no bundle** (`run_round` records only an `Idle` sink, 121-126) |
| `--classify` output | `reclassify` (415-430) | refuses any `format != FORMAT` (419-420) | derived verdicts |

Findings:

- **F5.1 The decision record is the unvalidated file.** Everything a user needs to
  answer "why" (deciding stage, reason, race, recovery, retries) lives only in
  `recycling.json`/evaluation outcomes, built from mutable dicts; the strictly verified
  artifact (the bundle) carries none of it. The two join only implicitly: by runtime
  `time_ns` (`_times`, 546-560), by scene revision (`"registration"`, 1036, 1097), and by
  execution ids inside the race basis (`tests/test_four_cans.py:76-79` reconstructs a
  timeline by joining `race.basis.during` to trace records). `recycling.json` names no
  bundle digest, so a moved outcome cannot be proven to belong to its bundle.
- **F5.2 `recycling.json` is not deterministic.** The embedded race report carries
  wall-clock `cost_s` and `duration_s` per plan (ssgrasping `candidates.py:360-400`,
  confirmed in the cutover evidence), and which lower-priority candidate was stopped is
  timing-dependent. `tests/test_sort_all.py:14-17, 40-50` already excludes the race from
  its determinism comparison. #246's "byte-identical after excluding no fields" is
  unachievable as stated.
- **F5.3 The race report is stored twice.** `race` = ssgrasping's summary `candidates`
  plus the raw ssrobot `ChainsReport` (`"schema": "ssrobot.ChainsReport", "version": 1`)
  under `chains` (ssgrasping `candidates.py:318-331`). In the cutover evaluation, `chains`
  is 1.33 MB of 5.50 MB (24%); `lookahead` logs 0.74 MB (14%); and each round's `objects`
  re-embeds every attempt already in `outcomes` (2.75 MB, 50%; `evaluate_recycling.py:
  312-330`, verified `objects[i].attempts[0] == outcomes[0]`).
- **F5.4 Output shape depends on CLI mode.** `--cans 1`: `{arm, **bin_outcome}` or
  `worktop_outcome` (783); `--cans N`: `{cans, succeeded}` (1322); `--mix`:
  `{attempts, objects, cleared, succeeded}` (1315). `scripts/check_installed.py:71-78`
  reads `report["checks"]`, so the installed-wheel gate depends on the one-can shape.
- **F5.5 Format churn without a reader.** The evaluator went `FORMAT` 3 → 6 in three PRs
  over ~1 day (`git log`: #252, #253, #254 after #250; docstring 67-83). The only frozen
  evidence, `evidence/cutover/20261031-070726-cutover.json.gz`, is format 3, so the
  current `--classify` refuses it (419-420). The "reproduce why" path for the claimed
  cutover result is therefore "check out 3bddbef", which the evidence README does state
  (`evidence/cutover/README.md`), but no code reads it.
- **F5.6 Reproducing a corpus round needs a re-run.** Evaluation rounds keep no bundle;
  the reproduction is `geodude-recycling --seed S --mix 5 --record DIR`, which matches
  `run_round` (`evaluate_recycling.py:113-126` vs recycling.py:1264-1314) but is not
  documented as the reproduction path, and the CLI does not refuse spent seeds (only
  `--round` does, 475-478). Uncertain whether a CLI rerun is byte-identical to a
  parallel evaluator round given planning timeouts (`_timed_out`, 184-194).
- **F5.7 The evaluator encodes protocol in strings.** `classify` returns
  `"no_arm_plans:..."`, `"checks:..."`, `"skill.stage:reason"` (197-221); `verdict` and
  `_summary` branch on `startswith("recycled")` (229-252). Essential content, string
  protocol.
- **F5.8 Diagnostics outside CI reach into private internals.** `scripts/time_startup.py:
  43-99` patches `ssgrasping._skill.Run.begin`, `ssrobot.mujoco._planning.native_mujoco`,
  and `recycling.warm_planning`; `scripts/calibrate_release.py:63-139` patches
  `ssgrasping._skill.Run.begin` and `ssrobot.mujoco._runtime.MujocoRuntime.step`. Neither
  runs in CI (`.github/workflows/ci.yml:92-117` runs only `check_boundary`, pytest,
  `check_installed`). mypy covers `scripts/` (pyproject.toml:61), which catches a
  removed attribute but not changed semantics. Tests also use privates:
  `tests/test_planning_latency.py:31-32` (`ssrobot._wire`, `ssrobot.planning._cbirrt`),
  `tests/test_registered_control.py:270-280` (`inner._model`, `_data`).
- **F5.9 Three copies of provenance**: `evaluate_recycling.py:333-355`,
  `calibrate_release.py:175-190`, `time_startup.py:142-156`; `recycling.json` records
  `sources` and kernel but not commit/dirty (1336-1342). Three small copies are
  acceptable by the repository's own rule; noted, not ranked.

## Scheduling and clock advancement

Five loops decide when the manual clock steps:

| Loop | Where | Rule |
|---|---|---|
| `run_together` | ssgrasping `execution.py:173-202`, via `drive` (349) and the race (1026) | step unless every unfinished item waits only on worker plans; then block |
| `_drop_racing_next` | recycling.py:1189-1220 | same rule re-implemented (1209-1214), plus: start an item when another reaches stage `transfer` (1206-1208); stop when the drop alone is done (1203) |
| `stepped` | 1005-1013 | not a loop: advances landings as a tick side effect after each step |
| `_wait` | 817-824 | step unconditionally (command executions; no worker plans) |
| final drain | 1145-1149 | step unconditionally until landings end |

Assessment: the essential rule ("never let planning latency pass simulated time", #241/#242,
proven by `tests/test_planning_latency.py:60-82`) is written twice (ssgrasping and
1209-1214). The two Geodude-specific needs that `run_together` lacks are (a) adding an
item when another reaches a stage and (b) finishing on a subset. Landings ride in the
tick callback instead of being items. The unconditional loops are correct (nothing plans)
but are separate code with separate exhaustion behavior (F4.5).

## Parallel, superseded, single-consumer paths

| Path | Consumers | Status |
|---|---|---|
| `demo` + `pick` + `drop_in_bin` (753-783, 355-367, 430-450) | `--cans 1` (1317-1319), `tests/test_recycling.py:67`; `pick` also `calibrate_release.py:155` | parallel to `recycle`; no race, no lookahead, no recovery, serial landing (450) |
| `place_on_worktop` + `worktop_outcome` (389-427, 634-684) | `demo --to worktop` only; `tests/test_recycling.py:77,115` | distinct task; keep (as #245 says) |
| `recycle(race=..., lookahead=...)` (947-948, 984-986, 794-795, 1043, 1052) | **no caller** in src, tests, scripts; `git log -S"race=False"` finds none ever | dead toggles "for comparison" |
| `--cans N>1` vs `--mix N` (1239-1258, 1313-1323) | `tests/test_four_cans.py:72`, `tests/test_planning_latency.py:43` vs `tests/test_sort_all.py:60` | two runners for one task, differing in retry policy and output shape |
| frozen-corpus checks in evaluator `main` (493-506) | unreachable: `FROZEN` (89) equals `SPENT`'s keys (91-94), refused first at 491-492 | dead until a new frozen corpus is declared |
| `warm_planning` (248-258) | `demo`, `recycle`; measured by `time_startup.py` | essential and tiny; keep |
| `scripts/time_startup.py` | one measurement for #224 | single-consumer diagnostic; patches privates |
| `scripts/calibrate_release.py` | derives the 16 mm sweep cited in `regions.py:167-170` | single-consumer calibration; the provenance of a model parameter |
| `_in_bin` (687-698) vs `regions.interior` (regions.py:231-247) | `bin_outcome`, `sort_all`, two tests | the same box defined twice |
| `checks.landed` vs `checks.in_bin` in `bin_outcome` (731, 733) | evaluator via `succeeded` | likely the same judgment at the same moment (uncertain: not measured; cutover evidence predates `landed`) |

## #245 assessment: one recycling execution path

Verdict: **confirm, narrow, and split**. Scope items:

1. *One-can bin demo as the one-object case of the runner* — confirmed by F4.1, F5.4 and
   the parallel-path table. But its acceptance criterion "existing development scenarios
   preserve ... recorded bundles" cannot hold for the one-can path: `recycle` plans a
   raced prefix through `Candidates` before moving, where `demo` calls `grasp` directly
   (771 vs 1025-1044), so the one-can trace will change (uncertain only in degree).
   Restate: four-can, latency, object-type, and `sort_all` bundles stay byte-identical;
   the one-can bundle and `recycling.json` shape change deliberately, with
   `check_installed.py:71-78`, `tests/test_recycling.py:98-118` and README lines 80-84
   updated. Deletes `demo`, `pick`, `drop_in_bin`, the serial landing, and the
   `--cans 1` branch (~70 lines, 4 functions); `calibrate_release.py` inlines
   `drive(grasp(...))`.
2. *`--cans` and `--mix` as staging choices of one runner* (not stated in #245; add it):
   both call `sort_all` over whatever was staged; `--cans` only picks `{"can": N}`. One
   output shape, one retry policy. Evidence: seed-4 four-can round succeeds at every
   first attempt (`tests/test_four_cans.py:82`), so adding `RETRIES=1` should not change
   its bundle (to be proven by digest).
3. *Exactly one component decides step vs wait* — confirmed (scheduling table). Narrow
   it: do it inside Geodude, by making `_drop_racing_next` express its two extra needs
   on top of a single loop, and by treating landings as items instead of tick side
   effects. Do **not** add "start-on-stage" or "finish-on-subset" to ssgrasping's
   `run_together` until ADA needs it (standing decision). Proof:
   `tests/test_planning_latency.py` plus byte-identical four-can and `sort_all` bundles.
   Uncertain: whether replacing the loop is exactly trace-preserving (iteration vs tick
   counting, F4.5); the digest comparison decides.
4. *Isolate orchestration from spec construction, outcome construction, and CLI* —
   narrow: outcome construction (542-750) becomes #246's records; CLI/viewer/report
   (1233-1352) is a natural `cli` boundary. Specs and constants can stay in
   `recycling.py`; a separate "policy" module would own nothing durable beyond data.

Order: #245 item 1-2 change the durable report shape; doing #246 first means the shape
changes once. Do #246's record core first, then #245. Neither depends on ADA as long as
the scheduler stays in Geodude.

## #246 assessment: typed, versioned recycling outcomes

Verdict: **confirm, narrow, reorder before #245**. Evidence: F4.2-F4.4, F5.1-F5.5.

Narrow:
- Own one Geodude record per attempt, recovery, final disposition, and run. Do **not**
  re-type the race report: embed ssrobot's already-versioned `ChainsReport` (or its
  digest) and drop ssgrasping's duplicate summary from the durable record, or the
  reverse (F5.3). The race-report shape is ssgrasping's to type, not Geodude's.
- Split wall-clock diagnostics (`cost_s`, `duration_s`, rest timing details in lookahead
  logs) from the decision record, or the byte-identical criterion must name them (F5.2).
- Make the in-progress attempt a private builder; serialize only completed records
  (F4.4). Make `safe_release`/`recover` read `Stage`, not the dict (F4.2).
- Record the bundle manifest digest in the run record (F5.1) and write atomically
  (`recycling.py:1343`; the evaluator already does, 399-404).
- Evaluator: drop `objects[].attempts` duplication (F5.3, ~50% of artifact) and type
  verdict/mechanism (F5.7).
- Decide explicitly whether old evaluation formats are read (F5.5). Recommendation: no
  migration framework; the frozen cutover evidence is reproduced by checking out its
  commit, as its README says.

What remains: every field #246 lists (stage, reason, evidence kind, held/attached, final
pose, checks, provenance), the classifications, and the frozen-corpus protocol. Proof: the
development corpus reclassified before/after with unchanged verdicts and summary;
tampered-copy refusals; two identical runs byte-identical.

## Ranked immediate deletions and consolidations

| # | Delete / consolidate | Behavior that remains | Proven by | Est. removed |
|---|---|---|---|---|
| 1 | `race` and `lookahead` parameters of `recycle` and `race` of `candidates` (947-948, 984-986, 794-795, 1043, 1052) | racing with lookahead, always | no caller (grep; `git log -S`); four-cans, latency, sort_all tests unchanged | 2 concepts, ~10 lines |
| 2 | `safe_release`/`recover` take `Stage` (+ a `no_zone` flag) instead of the `_failure` dict (849-927, 1135-1137) | identical release decisions | `tests/test_recovery.py` rewritten on `Stage` (same cases), dev-corpus recovery decisions unchanged | 1 dict-as-protocol hop |
| 3 | `_in_bin` re-expressed via `regions.interior` (687-698) | same final-bin judgment | `sort_all`/`registered_control` final bins, dev-corpus summary unchanged; uncertain whether sstsr `Tsr` offers containment | 1 duplicate concept, ~10 lines |
| 4 | Unreachable frozen-corpus branch in evaluator (`FROZEN`, 89-90, 493-506) | spent-corpus refusals | `SPENT ⊇ FROZEN` by inspection | ~16 lines; restore with the next frozen corpus (or keep as documented protocol, low priority) |
| 5 | `scripts/time_startup.py` (235 lines) once its result is recorded on #224 | `warm_planning` stays | its last artifact archived on #224 | 1 script that patches privates |
| 6 | One exhaustion policy for `MAX_TICKS` (F4.5) | same outcomes when budgets are not hit | four-cans/sort_all bundle digests | 3 divergent behaviors |

## Ranked larger candidates

| # | Candidate | Owner | Must remain | Removes | Evidence required |
|---|---|---|---|---|---|
| 1 | #246 narrowed: typed attempt/recovery/run records, builder for in-progress, bundle digest, atomic write, no duplicate race/attempts | geodude | all evaluator classifications; frozen protocol | 5 attempt shapes → 1; ~50-70% of evaluation bytes; string verdicts | dev corpus reclassified unchanged; tampered refusals; byte-identical reruns |
| 2 | #245 item 1+2: one runner (`sort_all`) for `--cans 1`, `--cans N`, `--mix`; worktop placement kept | geodude | one-can success, four-can racing evidence, retry policy | `demo`, `pick`, `drop_in_bin`, serial landing, two output shapes, two failure policies (~90 lines, ~5 concepts) | four-can/latency/sort_all/object-type bundles byte-identical; one-can shape change acknowledged |
| 3 | #245 item 3: one step-or-wait loop in Geodude (drop+next race, landings as items, drain) | geodude | #242 latency invariance | re-implemented rule (1209-1214), tick-side-effect landings, drain loop | `test_planning_latency`, bundle digests |
| 4 | Race report: one record, not summary + raw chains | ssgrasping (`candidates.py:318-331`) | selection, per-stage plan status used by `_timed_out`/`classify` | ~24% of evaluation bytes | evaluator unchanged on dev corpus |
| 5 | CLI module split (`main` 1233-1352) | geodude | installed command and documented outputs | nothing deleted; one boundary clarified | `check_installed.py` in CI |

## Do not generalize further

- Geodude's scheduling (`_drop_racing_next`, start-on-stage, landing items): stay in
  Geodude until ADA shows the same need.
- `recover`/`safe_release` and `RETRIES`: Geodude policy; no ssgrasping recovery API.
- The recycling outcome records (#246): Geodude-owned; no ssrobot/ssgrasping run-report
  format.
- `register`/`Read` registration seam (267-301): one consumer; do not lift into ssrobot
  until a perception source exists.
- `warm_planning`: a 10-line loop over an ssrobot call; no "bring-up" abstraction.
- Evaluator corpus/verdict machinery: Geodude script, not a framework.
- `calibrate_release.py`: one calibration; keep as provenance, do not build a calibration
  harness.

## Verdict: is Geodude 4.0 independently understandable?

Mostly yes for a user; not yet for a maintainer of failure handling and artifacts.

- A user can run and read it: the README is accurate (README.md:32-60, 61-117, 173-196),
  `scene.py` and `regions.py` are coherent boundaries, and every test writes a
  documented artifact.
- A maintainer is blocked by: (1) `recycle`'s 210-line loop holding race, grasp, drop,
  next race, landing callbacks, missed re-registration, recovery, and retry with mutable
  closures (941-1150); (2) the persisted dict being both the output and the recovery
  input (F4.2-F4.4); (3) three CLI modes with three failure policies and three report
  shapes (F4.1, F5.4); (4) a "why" that lives only in an unvalidated, non-deterministic,
  bundle-unlinked JSON whose format changed three times in a day (F5.1-F5.5).
- None of these needs ADA's input or an upstream abstraction. Recommended order: the
  immediate items 1-4, then #246 (narrowed), then #245 (narrowed), each proven by the
  existing behavioral tests and development-corpus digests.

Uncertainties: trace preservation of a unified driver; whether `checks.landed` and
`checks.in_bin` ever disagree; whether `sstsr.Tsr` has a containment test; byte-equality of
a CLI rerun with an evaluator round. None was executed in this audit.
