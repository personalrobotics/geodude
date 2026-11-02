# Accidental-complexity audit of the ssrobot / ssgrasping / Geodude 4.0 stack (#251)

This record is immutable. It measures fixed commits, and later changes are judged against
it, not folded into it.

| Repository | Audited snapshot | Baseline (last `main` commit before the Geodude 4.0 arc) |
| --- | --- | --- |
| ssrobot | `0ecbd8d` (after #232) | `116f3c8` |
| ssgrasping | `d29a4b4` (after #34, #28's estimates, #43) | `591f234` |
| geodude | `0618279` (after #248), legacy and `v4/` apart | `05f6359` |

The arc began with geodude `5efac0c` (2026-10-20), which started `v4/`. Every snapshot was
clean.

`#28` remains open for its `acquired` and `released` estimators. The snapshot follows its
`retained` estimator and its `placed`/`landed` recording, which is what #248 needed.

## Contents

- `inventory.py`: the quantitative inventory. It reads git objects only, so a working tree
  cannot change it; it uses the standard library only. Reproduce with:

  ```sh
  python3 inventory.py --ssrobot PATH --ssgrasping PATH --geodude PATH > inventory.json
  ```

  Each `PATH` is a clone containing the six commits. The run was recorded with Python
  3.14.7 and git 2.54.0. Two runs were byte-identical, and the output's SHA-256 is
  `2258f3001cd7b1277e0512c08e8c5643456940cc39201afb24e0e6a1f5059d44`.
- `inventory.json`: the machine-readable counts. It contains per-category lines and files,
  modules, classes, functions, public exports, the largest modules and functions,
  cross-repository imports, ssrobot schemas, and installed commands, at both commits.
- `findings/`: the seven area audits behind this synthesis, each cited by `file:line` at
  the snapshot commits. Three cover flows and public surface (ssrobot, ssgrasping,
  Geodude), and four cover tests (ssrobot contracts, ssrobot MuJoCo and artifacts,
  ssgrasping, Geodude `v4`). Each was done independently by reading the code. The two
  ssrobot test areas overlapped on five modules, and their findings agree where they
  overlap. The load-bearing claims were rechecked against the snapshot before inclusion,
  as section 9 lists.

**Categories, by path** (see `inventory.py`):

| Category | Paths |
| --- | --- |
| production | `src/` |
| tests | `tests/` |
| docs | Markdown, `docs/` |
| scripts | `scripts/`, `tasks/` |
| generated | ssrobot `schemas/`, `conformance/` |
| evidence | `evidence/` |
| examples | `examples/`, `references/` |
| config | everything else that is text |

- **Excluded:** lockfiles, binary files, and this directory.
- **Lines:** "code" lines are non-blank. In Python and shell files, comment-only lines
  are also excluded.

## 1. Quantitative inventory

Code lines by category:

| Part | When | production | tests | docs | scripts | generated | evidence | examples | config |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ssrobot | baseline | 14,561 | 10,906 | 2,828 | 1,631 | 13,795 | — | 1,124 | 200 |
| ssrobot | snapshot | 19,706 | 15,610 | 4,177 | 2,604 | 17,539 | — | 1,153 | 199 |
| ssgrasping | baseline | 1,023 | 420 | 69 | 83 | — | — | — | 131 |
| ssgrasping | snapshot | 2,939 | 1,688 | 710 | 733 | — | 10,142 | — | 137 |
| geodude legacy | baseline | 3,955 | 1,492 | 2,065 | 106 | — | — | — | 183 |
| geodude legacy | snapshot | 3,955 | 1,492 | 2,067 | 106 | — | — | — | 241 |
| geodude `v4/` | snapshot | 1,724 | 810 | 206 | 966 | — | 68 | — | 60 |

Combined production code (ssrobot, ssgrasping and Geodude 4.0) grew from 15,584 to 24,369
lines, about +56%. Legacy Geodude is unchanged, apart from `v4`'s CI jobs, which live in
the repository's root `.github/`.

| Part | Modules | Classes | Functions | Public exports | Test functions | Records (schemas) | Commands |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ssrobot | 44 → 48 | 212 → 268 | 708 → 880 | 144 → 190 | 444 → 672 | 39 → 49 | `ssrobot` |
| ssgrasping | 7 → 10 | 10 → 29 | 61 → 128 | 10 → 28 | 30 → 93 | none | none |
| geodude `v4/` | 0 → 4 | 0 → 3 | 0 → 75 | none | 0 → 35 | evaluation `FORMAT` 6, plus `recycling.json` (untyped) | `geodude-recycling` |

**Largest production units** (code lines):

| Repository | Module | Size | Responsibility |
| --- | --- | --- | --- |
| ssrobot | `context.py` | 2,706 | session and commands, registered-object scene, and the ~640-line plan-ahead/continuation state machine |
| ssrobot | `planning_contracts.py` | 1,526 | |
| ssrobot | `mujoco/_runtime.py` | 1,425 | |
| ssrobot | `chains.py` | 1,039 | |
| ssrobot | `portfolio.py` | 791 | the shared racing engine |
| ssgrasping | `_skill.py` | 870 | the `Run` class, ~600 lines, half of it planning-ahead machinery |
| ssgrasping | `grasp.py` | 452 | |
| ssgrasping | `place.py` | 394 | |
| ssgrasping | `candidates.py` | 375 | |
| geodude `v4/` | `recycling.py` | 1,199 | its regions are mapped in `findings/geodude-flows.md` |

The largest functions are ssrobot `execute_continuation` (249), `_open` (199) and
`predict` (160); ssgrasping `_grasp` (200); and Geodude `recycle` (210) and `main` (116).

**Cross-repository imports.** ssgrasping imports 66 distinct ssrobot names. Geodude 4.0
imports 72 distinct ssrobot and ssgrasping names. ssrobot imports neither. Two Geodude
scripts monkeypatch private ssgrasping and ssrobot symbols, which section 3 covers.

Growth is real, and most of it is essential. ssrobot's new code buys planning ahead,
continuation, rest evidence, guards, registered state, and the trust-boundary checks
around them, all exercised by the frozen corpus. The accidental part is concentrated, and
sections 2–6 name it.

## 2. Concept and ownership map: the five ordinary flows

Full traces, with every state transformation and serialization boundary, are in
`findings/`. The summary:

| Flow | Entry point → durable result | Owner | Concepts a caller holds | Accidental complexity found |
| --- | --- | --- | --- | --- |
| 1. Context, plan, prepare, execute | `RobotContext` → `ctx.plan` → `ctx.execute` → trace records | ssrobot | ~12 | **Coherent.** One owner and one record per step. There are three spellings of compile-and-submit, and ssgrasping's descent uses the superseded one. `run_until` has no consumer, because both consumers needed a per-tick hook. |
| 2. Race, then continue a plan made ahead | `ctx.start_chains`; `predict`, `project`, `continuation`, `watch_rest`, `execute_continuation` | ssrobot | +~15 | Three racing front ends run on one engine, and only chains has a consumer. Nine private session registries decide whether a plan may run. `Discrepancy.what` is prose that the consumer parses. `rebase` is not declared in the `Planner` protocol. Three representations of hypothetical state exist. |
| 3. Grasp, carry, drop or place, report the deciding stage | `grasp`, `drop`, `place` → `SkillResult` | ssgrasping | ~25 types plus string vocabularies | Six ways to plan a stage. Two feasibility definitions for one grasp. A superseded hand-off (`Follow`). Write-only `values` keys. Two declared but never-produced transitions. No owned serialization of `SkillResult`. |
| 4. Recover from a failed skill | `sort_all`/`recycle` → `_failure` → `safe_release`/`recover` → retry | Geodude | ~20 | Three failure policies selected by CLI flag. Recovery reads the serialized output dict. One attempt record comes in five shapes. One budget has four exhaustion behaviours. |
| 5. Inspect a run and reproduce its decision | bundle (verified) + `recycling.json` (unverified) + evaluation JSON | ssrobot, Geodude | — | The "why" lives only in an unvalidated, non-deterministic JSON that is not linked to its bundle. The evaluation artifact is about 50% duplicate. The format churned from 3 to 6 within about a day. |

**Ownership.**

The intended three-layer split holds:
- **ssrobot owns:** planning, execution, prediction, continuation, rest, and registered
  state.
- **ssgrasping owns:** stage contracts, feasibility prefixes, and carry estimates.
- **Geodude owns:** scheduling, recovery, assumption policy, and evaluation.

The misplacements are small:
- ssgrasping re-derives ssrobot race semantics (`canceled_by_priority`) from a JSON dump.
- ssgrasping reimplements ssrobot's default hand allowance.
- Geodude reimplements ssgrasping's step-or-wait rule.
- `landing`, a simulation-only evaluation, is shaped as a skill.

**Prose and dicts acting as protocol** (the main cross-layer coupling):
- `Discrepancy.what` is parsed by ssgrasping.
- `values["lookahead"]` prose is asserted in Geodude's tests.
- Stage-name strings are branched on by Geodude.
- The `Candidates.report` dict is parsed by Geodude's evaluator.
- The `_failure` dict is the recovery input.
- Evaluator verdicts are strings branched on by prefix.
- A private `Run.begin` hook is monkeypatched by Geodude scripts.

## 3. Public-surface classification

Every export added in the interval is classified in `findings/ssrobot-flows-surface.md`
§4 (46 names) and `findings/ssgrasping-flows-surface.md` §3 (18 names, and the 10 older
ones).

| Class | ssrobot (46 added) | ssgrasping (18 added) |
| --- | --- | --- |
| (a) two or more independent consumers | `GripperTermination`, `ObjectPoses` (borderline) | none can be: Geodude is the only consumer, and ssgrasping#6 (the two-robot proof) is open |
| (b) stable boundary, one consumer | 29, of which ~10 are public only because the closed-interface rule makes record parts public | `Selection`, `EstimateReason`, `TransitionEvaluation`, and, by Geodude use, `Candidate`, `Candidates`, `DropSpec`, `drop`, `Lookahead`, `Reason`, `Transition`, `TransitionEstimate`, `TransitionOutcome`, `run_together` |
| (c) application-specific, could move down | none | `landing`, `LandingSpec`: simulation evaluation |
| (d) compatibility-only | module-level `start_chains`, a second spelling of `ctx.start_chains` | none |
| (e) unsupported by demonstrated behaviour | 15: the nine alternatives names, `plan_chains`, `ObjectRest` (its consumer ssgrasping#36 is parked), `AttachmentReference` (recorded, no reader yet) | `Follow`, `Criterion`, `Observed` (carried, never read) |

Notes:
- **Gate:** most single-consumer additions fail ssrobot's own two-consumer gate
  (`docs/architecture.md`). For racing and continuation, the honest status is "one
  consumer, ssgrasping"; that is a boundary, not a generalization.
- **Documentation drift:** `README.md` says portfolios and alternatives are "used by
  ssgrasping and Geodude", which is false. `docs/contracts.md`'s public-name table omits
  four names added in #232 and lists `BASE_CONSTRAINTS`, which is not exported.

## 4. Test-to-contract map

Each test area was mapped test by test, and scenario by scenario for the large
end-to-end tests. Every finding states the failure mode protected, whether stronger
coverage exists, and whether the assertion is a change detector or reimplements
production logic.

| Area | Reviewed | Keep | Whole-test / run candidates | Assertion-level |
| --- | --- | --- | --- | --- |
| ssrobot contracts, planning, execution | 65 items, 20 files, 35 s | 59 | 5: schema test duplicates CI step; three folds into stronger siblings; one ownership test nearly duplicates conformance | 5 trims: hard-coded past versions, repeated conformance checks, a private call count |
| ssrobot MuJoCo and artifacts | 65 items, 23 files, 22 s | 51 (+2 partial) | 4 solid + 2 low-confidence: two authoring tests covered by the CLI test; a vacuous inference assertion (rewrite, not delete); a mostly subsumed package test | 10 trims: two tautologies, change detectors, an evidence field never asserted |
| ssgrasping skills and transitions | 30 scenarios, 34 MuJoCo sessions, 85–131 s in CI | 29 | 1 repeat run (determinism covered by `test_lookahead` and Geodude); `test_reference_robots` retires when the ADA slice lands, not before | 6 rewrites: asserted stage `detail` prose, a frozen temporary `land_ns`, a re-derived selection rule missing its tie-break, an absent-key check |
| Geodude `v4` application, evaluator, commands | 10 tests and the CI gates, ~6–6.5 min | all 10 | 4 repeat-only runs (~75–80 s), 1 duplicate CI step | 4: an exact ssgrasping sentence (`test_four_cans.py:98`), stage and name lists, a version check `check_installed.py` already does |

**No test area has many redundant tests.** Every module was written with its feature,
and most state their failure modes up front. The savings are:
- **Repeat-only runs.** Determinism is checked in many places. Declare one gate per
  layer: ssrobot's conformance `--check`, and Geodude's `test_planning_latency`.
- **Shared runs.** `test_four_cans` and `test_planning_latency` run the identical seed-4
  round. The same pedestal grasp is simulated in eight ssrobot modules.
- **Prose and incidental assertions.** These should become structured checks.

**Gaps, rather than redundancy:**
- Nothing in CI runs `evaluate_recycling.py`, whose `classify`/`verdict` produce the
  claimed corpus numbers.
- `test_planner_portfolio.py:517` accepts any refusal of a forged report instead of
  pinning the refusal codes.
- `test_recycling.py`'s docstring says CI runs it from the wheel; CI runs
  `geodude-recycling --seed 0`.

**Constraint on deletions.** ssgrasping's tests share private helpers by importing
across test modules, and `scripts/study_transitions.py` imports them too. Any test
deletion must keep or move those helpers, or the committed transition evidence stops
being reproducible.

No test is deleted in this audit. A deletion follow-up must name the failure mode, the
replacing coverage, and any unique boundary lost, as these findings do.

## 5. Ranked immediate deletions and consolidations

Each item is small and local, with no product decision, and none changes a demonstrated
behaviour. The ranking is by concepts removed per unit of risk.

| # | Owner | Delete or consolidate | Behaviour that remains | Proven by | Removes |
| --- | --- | --- | --- | --- | --- |
| 1 | ssgrasping | `Follow`, `Run.upcoming` and the `close` hand-off, `SkillExecution.upcoming`, `grasp(follow=)`, `drop(planned=)`, `transfer_request`. Superseded: ssgrasping#21 decided not to adopt it, and Geodude races with `Candidates(after=)`. | Geodude recycling, unchanged | Geodude `test_four_cans`, `test_sort_all`; ssgrasping `test_lookahead` minus its two `Follow` scenarios | 4 concepts, ~110 production and ~100 test lines |
| 2 | ssrobot | The alternatives racing front end: 9 exports, `ctx.start_alternatives`, its schema and docs. No consumer anywhere; chains express it. | Racing distinct requests as one-stage chains | `test_chains.py:225`, after porting three engine tests from `test_plan_alternatives.py` (`:461`, `:612`, `:759`) | 9 exports, 1 method, 1 schema, 1 outcome enum, 1 selection rule; ~870 production and ~730 net test lines |
| 3 | geodude | `recycle(race=, lookahead=)` and `candidates(race=)`: never passed by any caller | Racing with lookahead, always | `test_four_cans`, `test_planning_latency`, `test_sort_all` unchanged | 2 concepts |
| 4 | geodude | `recover`/`safe_release` read the typed `Stage` (plus a no-zone flag), not the `_failure` dict | Identical release decisions | `test_recovery` rewritten on `Stage`, same cases; dev-corpus recovery decisions unchanged | one dict-as-protocol hop |
| 5 | geodude | **Bug:** if the final landing drain exhausts `MAX_TICKS`, an attempt record escapes without `succeeded`, and `main`'s `--cans` branch raises `KeyError`. Serialize only completed records. | Unchanged when budgets are not hit | A focused test of the exhaustion path, or #246's builder | one partial shape |
| 6 | ssgrasping | `GraspSpec.deepest`, `approach_tries`, `_deepest_approach`, `values["approach_m"]`: parked since geodude `71e31c8` | Approach at `approach_m`, as Geodude runs | Geodude corpus; ssgrasping `test_grasp_place` | 1 concept, ~65 lines |
| 7 | ssgrasping | `place._descend`'s hand-built `prepare` + `submit` → `ctx.execute(plan, guard=)` (available since ssrobot#228) | The guarded descent, now also traced `compiled` | ssgrasping `success`/`no_contact`; Geodude worktop test | one parallel execution path; afterwards ssrobot's free `prepare` export loses its consumer |
| 8 | ssgrasping | Write-only `values` keys `closed_opening`, `lifted_opening`, `hand_allowance`, `contact_pose`, `descent`; `Transition.ACQUIRED`/`RELEASED` (never produced); `Candidates(releases=)` (superseded by `removals`) | All behaviour | grep shows no readers or producers | 5 protocol keys, 2 enum members, 1 parameter |
| 9 | ssrobot | Typed `Discrepancy` fields (kind, subject) in place of parsed `what`; `rebase` declared in the `Planner` protocol; `plan`/`plan_async` result check written once | Same refusals | `test_continuation.py:698`, `test_offset_continuation.py:300` | two undeclared protocols (`ContinuationReport` bumps a version) |
| 10 | ssrobot, ssgrasping | Documentation: README racing claims, contracts.md public table, stale `chains.py` prose; README notes that `run_skill` lacks `tick` | — | — | false concepts a reader would otherwise learn |
| 11 | geodude | Unreachable frozen-corpus branch in the evaluator (`FROZEN` ⊆ `SPENT`); `_in_bin` re-expressed through `regions.interior` | Same refusals and bins | Inspection; dev-corpus summary unchanged | ~25 lines, one duplicate definition |
| 12 | all | Tests: drop repeat-only runs; declare one determinism gate per layer; share the seed-4 run; replace prose and selection-rule assertions with structured checks; add the evaluator to CI | Every protected failure mode in section 4 | The section 4 maps | ~2 min of CI, 10+ incidental assertions |

## 6. Ranked larger candidates

These need a decision, or proof beyond the existing gates.

| # | Candidate | Owner | Must remain | Est. removed | Evidence required |
| --- | --- | --- | --- | --- | --- |
| A | **#246, narrowed and moved before #245.** Geodude-owned typed records for attempt, recovery and run: a builder for in-progress state, the bundle digest, an atomic write. Embed ssrobot's `ChainsReport` instead of re-typing the race; split wall-clock fields from the decision record; drop the evaluator's duplicate `objects[].attempts`. | geodude | every evaluator classification; the frozen-corpus protocol | 5 attempt shapes → 1; ~50–70% of evaluation bytes; string verdicts | dev corpus reclassified with unchanged verdicts; tampered-copy refusals; byte-identical reruns |
| B | **#245, narrowed.** One runner (`sort_all`) for `--cans 1`, `--cans N` and `--mix` (one retry policy, one output shape), with worktop placement kept separate. One step-or-wait loop inside Geodude: landings become items, and `_drop_racing_next` sits on the single loop. | geodude | one-can success; four-can racing evidence; the #242 latency invariance | `demo`, `pick`, `drop_in_bin`, the serial landing, 2 output shapes, 2 failure policies, the duplicated step rule (~90 lines, ~5 concepts) | four-can, latency, object-type and `sort_all` bundles byte-identical. The one-can bundle and report change deliberately, so `check_installed.py` and the README are updated. |
| C | **One racing engine.** With alternatives gone (item 2), decide whether portfolios stay a user feature. If not, `_JobRun` moves into `chains.py`, ending private cross-module imports and the engine's writes into context-private caches. | ssrobot | chains' selection and refusal contracts | ~340 lines, 7 exports, 1 schema | `test_chains`; ssgrasping `test_lookahead` |
| D | **One planning-ahead record in ssgrasping.** Replace `values["lookahead"]` prose with a typed entry or the ssrobot trace, keep only dynamic boundaries, and serialize ssrobot records with `dumps` instead of hand-flattening. | ssgrasping | Geodude's timeline cross-check (`test_four_cans`) | ~150 lines of `Run`; one untyped cross-repo protocol | Geodude `test_four_cans` against the trace |
| E | **Typed, versioned `SkillResult` serialization owned by ssgrasping**, replacing the open `values` dict. | ssgrasping | every field Geodude persists | Geodude's four ad hoc flatteners; the tests' `_report` | ssgrasping and Geodude artifacts unchanged in content. Coordinate with A. |
| F | **One feasibility definition per skill.** A standalone `grasp` (and a transfer) as a one-candidate chain, retiring `Run.choose`'s seed loop. | ssgrasping | standalone grasp and place behaviour | one of two planners for the same prefix | **Cross plans** before attributing any outcome change (seed changes shift later CBiRRT plans) |
| G | **Move the continuation state machine out of `RobotContext`** (`predict` … `execute_continuation`, ~640 lines, five registries) into `continuation.py`. | ssrobot | every continuation contract | nothing deleted; `RobotContext` becomes "session and commands" | the whole continuation suite |
| H | **Decide the LIVE object-pose mode.** Geodude controls only with REGISTERED; LIVE is the default and is used by ssrobot and ssgrasping tests. If LIVE becomes an evaluation-only mode, the two-regime branches go. So does ssgrasping's `replace_allowance` refusal on the simulator's attachment monitor: it is inert under REGISTERED, and it is the one remaining privileged control branch (#248), live mode only. | ssrobot, ssgrasping | REGISTERED control; LIVE observation for evaluation | two-regime cross product; one privileged branch | ssrobot `test_registered_scene`; ssgrasping `drifted` rewritten |
| I | **`landing` as an evaluation, not a skill**, once ssgrasping#36 waits on observed rest. It becomes `watch_rest(ObjectRest)` plus a pure scoring function, which also gives `ObjectRest` its consumer. Until then, keep both. | ssgrasping | the landed evaluation | `Wait.until_ns`, a `Run` and `Stage` that command nothing, `Reason.STILL_ATTACHED` | Geodude assumption counts unchanged |

`AttachmentReference` (#232) was added by explicit review request as the evaluation
record of declared-vs-tracked transforms. It has no reader yet. It stays until the
evaluator either reads it or the decision is reversed; it is not a deletion candidate on
usage alone.

## 7. Do not generalize further

These have one demonstrated consumer. Hold them where they are until ADA, or another
consumer, needs the same thing:

- **ssrobot chains:** no stage policies beyond `seeds` and `retry_prefix`. `abort=` and
  `cancel_lower` are unused by the consumer.
- **`predict`'s declared-change parameters** (`releases`, `removals`, `allowances`,
  `joints`, `unsettled`) and `execute_continuation`'s object-tolerance knobs. These are
  driven by ssgrasping alone. After deletions 1 and 8, `allowances` and `releases` lose
  their consumer, so re-check them then.
- **`OpeningGuard`, registered objects, caller-supplied `RestCriteria`.**
- **ssgrasping `Candidates`:** grasp plus an optional drop only. No place prefix and no
  N-skill chains.
- **The `Lookahead` knobs:** Geodude uses only `unsettled`.
- **The `released` callback.**
- **`Observed` and `Criterion`.**
- **`landing`.**
- **The `Candidates.report` semantics.**
- **Geodude's scheduling** (`_drop_racing_next`, start-on-stage, landing items):
  standing decision.
- **Recovery and `RETRIES`.**
- **The recycling outcome records.**
- **The `register`/`Read` registration seam.**
- **`warm_planning`.**
- **The evaluator's corpus and verdict machinery.**
- **`calibrate_release.py`:** keep as the provenance of the 16 mm sweep, with no
  calibration harness.

## 8. #245 and #246

- **#246: confirmed, narrowed, and ordered first.**
  - **Why it is needed:** recovery reads the serialized dict; one record comes in five
    shapes, and one can escape incomplete; the "why" record is unvalidated, not
    deterministic, and not linked to its bundle; the evaluation artifact is half
    duplicate.
  - **The narrowing is in A.** Its "byte-identical after excluding no fields" criterion
    cannot pass while race reports carry wall-clock `cost_s`/`duration_s`, so restate it
    with those fields split out.
- **#245: confirmed, narrowed, and ordered after #246**, so the durable report shape
  changes once.
  - **Add:** `--cans N` and `--mix` become staging choices of one runner, which #245 does
    not state today.
  - **Restate:** its "preserve recorded bundles" criterion, for the one-can path, whose
    planning changes when it races.
  - **Keep the step-or-wait unification inside Geodude**, by the standing decision.
  - **Neither issue needs ADA.**

## 9. Verification of load-bearing claims

The claims below were rechecked at the snapshot before inclusion:
- **No consumer outside ssrobot uses the alternatives or portfolio names.** ssgrasping
  uses only `ctx.start_chains`.
- **Geodude uses none of** `Follow`, `upcoming`, `transfer_request`, `planned=` or
  `follow=`, and no caller passes `recycle(race=, lookahead=)`.
- **No consumer uses `GraspSpec.deepest` or `approach_tries`.**
- **The three failure policies:** `demo` has no recovery, the `--cans N` branch calls
  `recycle` with `retries=0`, and `--mix` retries once.
- **The final landing drain can exhaust `MAX_TICKS`** and leave an entry without
  `succeeded`, which `main`'s `--cans` branch then indexes.
- **`place._descend` hand-compiles** with `prepare` and `ctx.submit`.

**Not executed in this audit, and to be proven by the follow-ups:**
- whether a unified driver preserves traces exactly;
- whether `checks.landed` and `checks.in_bin` ever disagree;
- whether a CLI rerun is byte-identical to an evaluator round;
- whether the descent's switch to `ctx.execute` changes any committed digest.

## 10. Recommendation

**Independently understandable?**
- **ssrobot: yes for flow 1, not yet for flow 2.**
  - The surface advertises mechanisms with no consumer.
  - Hidden registries and undeclared protocols decide validity.
  - One 2.7k-line class holds three subsystems.
  - Its essential contracts are sound and well tested.
- **ssgrasping: partly.**
  - The stage contract reads on its own.
  - Independence is blocked by one robot (ssgrasping#6 open), Geodude-shaped and parked
    features, dict and prose protocols shared with Geodude, and no owned outcome
    serialization.
- **Geodude 4.0: yes for a user, not yet for a maintainer of failure handling and
  artifacts.** The causes are the 210-line `recycle` loop, the dict that is both output
  and recovery input, the three CLI modes, and the unchecked "why" record.

**Order: a bounded cleanup first, then the ADA slice and Geodude's records in parallel.**

1. **Bounded cleanup.**
   - The immediate list in section 5, in the repository that owns each item.
   - Larger candidate C, if portfolios are not kept as a user feature.
   - Decision H on the LIVE pose mode.

   This removes about 20 concepts and roughly 1.5k production and test lines. Most of
   it is surface ADA would otherwise have to learn: alternatives, `Follow`, the parked
   grasp depth, and the undeclared protocols.
2. **Then, in parallel:**
   - **The ADA slice** (ssgrasping#6, gated by ssrobot#121 for jointed tools). Hold
     everything on the "do not generalize further" list until ADA demands it.
   - **Geodude:** #246 narrowed, then #245 narrowed. Both are Geodude-only.
3. **Later, when a second consumer or the ADA slice gives evidence:** candidates D, E, F,
   G and I.

The stack is ready for the ADA slice once step 1 lands: no essential contract needs
rework first, and the cleanup only removes surface. Implementation issues are opened per
repository only after this record is accepted.
