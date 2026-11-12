# ssgrasping production code, flow 3, and public surface (geodude#251)

Snapshot: ssgrasping d29a4b4, geodude 0618279 (`v4/`), ssrobot 0ecbd8d. Baseline ssgrasping
591f234. Read-only. Paths below are relative to each worktree under
a worktree at each snapshot commit. "SG" = ssgrasping, "G" = geodude `v4/`.

Size: `src/ssgrasping` went from 1,203 to 3,320 lines (`wc -l`; baseline per file via
`git show 591f234:<f> | wc -l`). `_skill.py` 296 → 960, `grasp.py` 228 → 503, and three
new modules, `candidates.py` (422), `drop.py` (305) and `landing.py` (94). Public exports
went from 10 to 28 (`src/ssgrasping/__init__.py:31-60` against `git show 591f234:src/ssgrasping/__init__.py`).

**Consumers.** Geodude v4 is the only consumer outside SG. A search of `~/code` for
`from ssgrasping`/`import ssgrasping`, excluding SG, its baseline copy and the geodude
worktrees, found nothing. Every SG behaviour test runs on Geodude only:
`tests/test_grasp_place.py:368`, `tests/test_lookahead.py:228` and
`scripts/study_transitions.py:299,385` load `geodude_assets`. ADA is only *opened* in
`tests/test_reference_robots.py:43`. The two-robot proof (ssgrasping#6) is **OPEN**
(`gh issue view 6`). So no export has two *independent* consumers. Category (a) below
means "Geodude, plus SG's own behavior tests".

---

## 1. Flow 3: grasp, carry, drop or place, and the deciding stage

### Entry points as Geodude runs them

`recycle` (G `src/geodude/recycling.py:940-1155`):
1. `Candidates(ctx, [Candidate(...)], unsettled=, after=, removals=, joints=)` (1025, 1069-1076)
2. `run_together(ctx, [pending])` (1026)
3. `selection.grasp(ctx, lookahead=Lookahead(unsettled=))` (1043-1044), driven by G's own `drive` (344-352)
4. `drop(ctx, planner, held_drop_spec(...), check_planner=, lookahead=, released=assume_landed(...))` (1053-1060), driven by G's own `_drop_racing_next` (1189-1222)
5. `landing(ctx, landing_spec, _landed(dropped))` (1114), advanced inside G's `stepped` tick (1005-1012)
6. G turns the three `SkillResult`s into dicts with `_failure`, `_transitions`, `_stages` and `_times` (542-623)

`place` is used only on the single-object worktop path (`place_on_worktop`, 389-427).

### Concepts and public types a caller must understand

There are about 25 types, plus string vocabularies:

- **Specs:** `GraspSpec` (17 fields), `DropSpec` (16), `PlaceSpec` (22), `LandingSpec` (4).
- **Two planners per skill**, `planner` and `check_planner`.
- **Planning ahead:** `Lookahead`, `Candidate`, `Candidates`, `Selection`, and `Follow` (unused by G).
- **Driving:** `SkillExecution`, whose members are `advance`, `cancel`, `stage`, `active`, `planning`, `upcoming`, `result` and `done`; `run_skill`; `run_together`.
- **Results:** `SkillResult`, with `stages`, `values`, `transitions`, `evaluations` and `attachment`. `Stage` has 10 fields.
- **Enums:** `SkillState` (5 members), `Reason` (17), `Evidence` (4), `Transition` (5), `TransitionOutcome` (3), `EstimateReason` (4).
- **Estimates:** `TransitionEstimate`, `TransitionEvaluation`, `Observed`, `Criterion`.
- **The `released` callback** (`drop.py:138`, `place.py:176`).

The string vocabularies are:
- **Stage names:** `"release"`, `"transfer"`, `"lift"`, … They are only documented in module docstrings (`grasp.py:5-49`, `drop.py:7-25`, `place.py:6-37`).
- **`values` keys** (§1.6).
- **Lookahead-log keys and "decision" phrases** (§1.6).
- **The `Candidates.report` dict** (`candidates.py:328-334`, 388-406).

### State transformations

1. **Spec construction.** Each spec is validated in `__post_init__` through `check_spec` (`_skill.py:949-952`; for example `grasp.py:147-161`).
2. **`Run`** (`_skill.py:266-290`) is a *mutable* accumulator: `stage`, `stages`, `values`, `held`, `transitions`, `evaluations`, `_aheads` and `upcoming`.
3. **The skill body is a generator** that yields `Wait` (`_skill.py:72-103`). `Wait` takes exactly one of `execution`, `until_ns`, `future` or `watch`. `SkillExecution.advance` (`execution.py:75-95`) resumes the generator while `Wait.satisfied`.
4. **Failure is control flow.** `Run.record(..., succeeded=False)` appends a `Stage` and raises `Failed` (`_skill.py:403-417`). `advance` catches `Failed` and `SsrobotError` (`execution.py:87-94`).
5. **Scene mutation goes through ssrobot:**
   - `ctx.attach` (`grasp.py:341`);
   - `Run.replace_allowance`, which detaches and re-attaches (`_skill.py:777-811`);
   - `ctx.detach` (`drop.py:280`, `place.py:303`);
   - then the application's `released` callback (`drop.py:284-285`), which in G registers or removes the object (`assume_landed`).
6. **Ending.** `_finish` → `Run.close` (`_skill.py:379-394`) cancels unused aheads or hands one on. `Run.result` then freezes everything into `SkillResult` (`_skill.py:436-445`).

### Serialization boundaries

- **ssgrasping owns no serialization of `SkillResult`.** The outcome types are plain dataclasses (`outcomes.py:118-216`), not ssrobot `Record`s, so they have no schema version. Each consumer flattens them its own way:
  - SG tests: `_report` (`tests/test_grasp_place.py:312-350`);
  - G: `_failure`, `_transitions`, `_stages` and `_times` (`recycling.py:542-623`);
  - G's evaluator then reads those dicts (`scripts/evaluate_recycling.py:185-221, 278-298`).
- **Inside `values`, ssrobot records are serialized two different ways.**
  - `GripperResult` goes through ssrobot `dumps` (`_skill.py:725`).
  - `RestResult` (a `Record`, `ssrobot/src/ssrobot/rest.py:106`) is hand-flattened by `rest_evidence` (`_skill.py:242-259`).
  - `PredictedSnapshot` (a `Record`, `continuation.py:82`) is hand-flattened by `lookahead_entry` (`_skill.py:223-239`).
  - `ContinuationReport` (a `Record`, `continuation.py:234`) is hand-copied into the log (`_skill.py:640-646`).
- **The only typed, versioned durable record of planning ahead is ssrobot's trace.** G's `test_four_cans` cross-checks the SG lookahead log against the trace's `PREDICTED` and `CONTINUED` records (G `tests/test_four_cans.py:37-65, 109-111`).

### Ownership

**Clearly ssgrasping's:**
- stage sequencing and the stage contracts;
- feasibility prefixes;
- carry estimates (`retention`, `_retained`, `_skill.py:157-176, 536-583`);
- `preflight`.

**Clearly ssrobot's**, and correctly delegated: planning, execution, prediction, continuation, rest, chains.

**Clearly the application's:**
- the `released` policy;
- treating transitions as assumed (G `recycling.py:563-566`);
- recovery (G `recycling.py:851-862`);
- driving the clock.

**Questionable:**
- **`landing`** (`landing.py`) is a simulation-only evaluation dressed as a skill. It has its own `Run`, `Stage` and `Reason`s (`STILL_ATTACHED`, `UNAVAILABLE`), and is the only user of the temporary `Wait.until_ns` (`landing.py:74`).
- **`Candidates`** hard-codes Geodude's recycling prefix: grasp, then lift, then drop into a container (`candidates.py:197-298`). `place` has no candidate form.
- **`_hand_frames`** (`candidates.py:409-422`) reimplements "the frames ssrobot allows a held object to touch by default". `grasp` instead reads ssrobot's own default back (`grasp.py:339-344`). That is two sources for one ssrobot rule.

### Duplicated mechanisms

**Plan**

There are six paths:
1. `Run.choose`: serial seeds, `ctx.prepare`, then a follow-on check (`_skill.py:813-865`).
2. `Candidates`: ssrobot `start_chains` (`candidates.py:159-161`).
3. `Run.ahead`: on a prediction (`_skill.py:303-333`).
4. `Run.kept`: a plan made on a projection, bound as a continuation (`_skill.py:335-370`).
5. `Follow` with `Run.upcoming` (`_skill.py:190-200`, `grasp.py:358-364`).
6. Replan closures (`grasp.py:246-298`), plus `_deepest_approach` (`grasp.py:432-455`).

The same skill gets two different feasibility definitions:
- `grasp()` checks pregrasp→approach (`grasp.py:246-263`).
- `Candidates` checks pregrasp→approach→lift→transfer(→retreat) (`candidates.py:197-298`).
- `Run.choose` additionally rejects `execution_unready` (`_skill.py:846-852`). ssrobot chains validate but are not shown to `prepare`; this is uncertain, from `ssrobot/src/ssrobot/chains.py:4-14`.

The drop's transfer request is written three times:
- `drop.py:235-243` (`choose`);
- `drop.py:179-186` (`transfer_request`);
- `candidates.py:243-262`.

The grasp's lift request is written twice (`grasp.py:354-356`, `candidates.py:219-237`).

**Execute**

`place._descend` hand-compiles with `prepare(...)`, builds `GuardedJointTrajectory` and calls `ctx.submit` (`place.py:349-358`). `ctx.execute(plan, planner=, guard=)` now accepts any `Guard = WrenchGuard | OpeningGuard` (`ssrobot/src/ssrobot/context.py:1931-1960`, `commands.py:317`). That arrived in ssrobot#228 (e27a25a), after `_descend` (9d49ece, 2026-10-17). The path is superseded by history.

**Wait**

- the four `Wait` kinds;
- `Run.rest` (`_skill.py:664-672`);
- `Lookahead.rest_timeout_ns` *and* `PlaceSpec.rest_timeout_ns`, two homes for the same `REST_TIMEOUT_NS` (`_skill.py:137`, `place.py:129`);
- G's own `_wait` (`recycling.py:996-1003`).

**Advance time**

In SG, `run_skill` (`execution.py:125-158`) and `run_together` (`execution.py:173-202`).

In G, four more drivers:
- `drive` is `run_together` plus `cancel`, the same as `run_skill`'s manual branch but with `tick` (`recycling.py:344-352`);
- `_drop_racing_next` reimplements `run_together` so that it can add a race mid-run (`recycling.py:1189-1222`);
- the landings advanced inside `stepped` (`recycling.py:1005-1012`);
- the trailing loop (`recycling.py:1150-1154`).

SG's own test does the same "race next" job with plain `run_together(..., tick=start_next)` (`tests/test_lookahead.py:442-464`).

**Record failure**

There are seven call sites that create a failed `Stage`:
- `Run.record`;
- `Run.fail`;
- `Run.cancelled`;
- `_execute`;
- `_replanned`;
- `finished`;
- `preflight` (`_skill.py:403-430, 585-604, 674-711, 886-946`).

The lookahead log records the same failure *again*, as a prose `decision` such as `"failed: for recovery"` or `"replanned: …"` (`_skill.py:619, 651, 659`).

**Inspect state**

- `SkillExecution.stage`, `.active` and `.planning`;
- `Stage.held` and `.attached`;
- `SkillResult.attachment`;
- `values`;
- the lookahead log;
- the ssrobot trace.

G scripts monkeypatch the private `ssgrasping._skill.Run.begin` to see stage starts (G `scripts/calibrate_release.py:63,78,138`; `scripts/time_startup.py:43,79,86`). That is a private API acting as an observation hook.

**Produce an outcome**

- `SkillResult.state` and the deciding `Stage`;
- `TransitionEstimate`;
- `TransitionEvaluation`;
- a separate `landing` `SkillResult` whose "success" means *scored*, not *landed*. G must then join evaluations across results by `(stage, transition)` (`recycling.py:578-583`), even though an evaluation already carries its estimate;
- `Candidates.report` (a dict) plus `selection`.

**"Released" has three representations:**
- the stage name `"release"`, read by G `_released` (`recycling.py:626-627`) and `_times` (554);
- `Transition.RELEASED`, declared but **never estimated** (`outcomes.py:85`; there are no producers in `src`);
- the `released` callback.

`Transition.ACQUIRED` is likewise only a *stage*, never an estimate (`outcomes.py:81`, `grasp.py:375-392`).

### Prose, dicts or implicit state acting as a protocol

1. **`values["lookahead"]`** is a list of untyped dicts (`_skill.py:223-239, 297-301, 325, 361, 364, 383-394, 619-659`). G persists it as `entry["lookahead"]` (`recycling.py:1087-1098`). G's tests assert on its **prose**:
   - `approach.get("kept") == "the feasibility check, planned before"` (G `tests/test_four_cans.py:98`);
   - `pregrasp["decision"] == "continued"` (`:111`).

   SG tests assert `"handed on: continued by the next skill"` (`tests/test_lookahead.py:704`). This duplicates ssrobot's typed trace records (§1.4).
2. **Stage `detail` is asserted in tests**: `"succeeded (planned ahead, continued)"` (`tests/test_lookahead.py:707`) and `"replanned: arrival_out_of_tolerance" in detail` (`:722`). Yet `outcomes.py:9-11` says the detail "may be reworded at any time".
3. **The evidence kind is decided from prose.** `any(d.what.startswith("object ") for d in report.discrepancies)` (`_skill.py:652-656`).
4. **Stage names are a string protocol** read by G:
   - `dropping.stage == "transfer"` (`recycling.py:1206`);
   - `("grasp", "lift", Reason.NOT_PLANNED)` (`:859`);
   - `s.name == "release"` (`:627`, `:554`).
5. **The `SkillResult.values` keys** (`outcomes.py:202-203`, documented only as "measured values worth keeping"):

   | Key | Written at | Readers |
   |---|---|---|
   | `lookahead` | `_skill.py:293` | **G** (`recycling.py:1088`), G test (`test_four_cans.py:94-111`), SG tests |
   | `gripper` | `_skill.py:724-725` | SG test (`test_grasp_place.py:685,694`), `study_transitions.py:443` |
   | `attach_transform` | `grasp.py:345` | SG test (`test_grasp_place.py:612`) |
   | `tare_rest` | `place.py:327` | SG test (`test_grasp_place.py:562,621`) |
   | `approach_m` | `grasp.py:289,295` | SG test (`test_lookahead.py:818`), only with `deepest` |
   | `released_opening` | `drop.py:270`, `place.py:293` | SG test asserts it is *absent* (`test_drop.py:398`) |
   | `closed_opening`, `lifted_opening`, `hand_allowance`, `contact_pose`, `descent` | `grasp.py:317,377,350`; `place.py:277,363` | **none** (write-only) |
   | `<stage>_attempts` | `_skill.py:840` | README table only (`README.md:61`) |

   Geodude reads exactly one key, `lookahead`. Every other key is dropped from G's durable artifacts. `closed_opening` repeats `gripper["close"]`'s opening, the `held_opening` `Criterion` (`grasp.py:368`) and the stage detail (`grasp.py:318`).
6. **`Candidates.report`** is a dict (`candidates.py:328-334`) holding both a re-derived `candidates` summary and the raw ssrobot `chains` report. The summary re-derives `canceled` and `canceled_by_priority` from ssrobot's report fields (`candidates.py:360-406`), so ssgrasping is reinterpreting ssrobot semantics. G's evaluator reads `race.candidates[].plans[].stage` (G `scripts/evaluate_recycling.py:191,206-207`).
7. **Aliasing trick in `Run.close`.** It writes `handed.log["decision"]` into the shared dict, then rebinds `handed.log` to a copy without it (`_skill.py:383-387`). The grasp's log keeps the decision while the drop's adopted entry does not.
8. **Privileged state decides a stage.** `replace_allowance` fails the stage with `ATTACHMENT_LOST` when `attachment.held` is False (`_skill.py:792-799`). ssrobot sets that flag only from its monitor ("False once the runtime reported the object left the grasp", `ssrobot/src/ssrobot/scene.py:44-46`). That is a privileged branch in control, which is at odds with geodude#248's "no privileged control" and with `outcomes.py:14-18`.

   Uncertain: whether a physical runtime can ever set `held=False`. If it cannot, the branch is dead off-simulator; in simulation it is privileged control.

---

## 2. Concentration points

| Point | Verdict | Evidence |
|---|---|---|
| `Run` (`_skill.py:266-865`, about 600 lines) | Essential core, but too much in one class: stage recording, five planning-ahead paths, `choose`, `rest`, `grip`, `evaluate` and `replace_allowance` | Every skill uses it. The planning-ahead half (`ahead`, `kept`, `adopt`, `close` hand-off, `upcoming`, `_continue`, `_replanned`) is about 250 lines |
| `Wait` (`_skill.py:72-100`) | Essential: it is what makes skills passive and lets drivers block on plans without stepping. `until_ns` exists only for `landing`'s temporary fixed wait (`landing.py:11-13, 47-51`; ssgrasping#36 OPEN) | `execution.py:58-65`, `execution.py:189-199` |
| `Ahead` | Essential as the "plan made on a prediction" handle, but overloaded. It is built four ways: `Run.ahead`, `Run.kept` (with a pre-bound `continuation`), `Candidates._select` (a synthetic done `Future`, `candidates.py:340-351`), and `Follow` | `_skill.py:203-220` |
| `Kept` | Essential: a projection-made plan that becomes an `Ahead` only once its source runs. It has one producer pair (`grasp.choose`, `Candidates._select`) and one use (`approach`) | `_skill.py:179-187`, `grasp.py:262,274`, `candidates.py:356` |
| `Follow`, `upcoming`, `drop(planned=)`, `transfer_request` | **Accidental; superseded.** Geodude decided *not* to merge the wiring: it gained about 0.1 s per round of about 31.6 s and lost first-attempt successes (ssgrasping#21 decision comment). G uses `Candidates(after=transfer)` instead (`recycling.py:1067-1076`). Its only consumer is SG's `transfer_ahead` and `transfer_disturbed` scenarios | grep: `Follow` appears only in `tests/test_lookahead.py` |
| `Run.move` paths (`_skill.py:474-534`) | Three branches: `Ahead`→`_continue`; a `PlanResult` with `replan` (execute, or replan on `start_mismatch`); a plain `PlanResult`. The middle branch is essential (a race or choice made on a snapshot the robot may have left) | `grasp.py:302-308`, `drop.py:263` |
| Stage recording (`begin`, `record`, `fail`, `cancelled`, `_append`, `boundary`) | `begin`, `record` and `_append` are essential. `boundary` logs a fact that is **static per skill** (always `close`, `restrict`, `release`; asserted at `tests/test_lookahead.py:690-692`, G `test_four_cans.py:99`) and is already in the docstrings (`grasp.py:55-56`). Its dynamic cases (`unpredictable:`, `kept plan refused:`) are the useful part | `_skill.py:297-301, 325, 361` |
| `Reason` (17 members) | Essential and stable. All 17 are produced. G branches on `MISSED`, `NOT_PLANNED` and `TRANSITION_REFUTED` (`recycling.py:859-861,1122`). The evaluator reads `missed`, `lost` and `not_planned` (`evaluate_recycling.py:189,217-219`). Minor: `CANCELLED` and `COMMAND_CANCELED` are spelled differently (`outcomes.py:63-64`) | — |
| `TransitionEstimate`, `TransitionEvaluation` | Essential for #28 and #248: G's assumption policy and evaluator use the outcome, the reference and the source (`recycling.py:573-603`). `Observed` and `Criterion` are carried but **never read** by any consumer except through `asdict` in an SG test (`test_grasp_place.py:340`) | — |
| `Lookahead` | Essential (opt-in planning ahead). G uses only `unsettled`. `arrival_tolerance` and `rest_timeout_ns` serve only the SG `mismatch` and `not_at_rest` scenarios (`tests/test_lookahead.py:342,618`) | `_skill.py:127-153` |
| `Candidates` and `Selection` | Essential to G (the arm race). Geodude-shaped, so do not generalize. `releases=` is superseded for G by `removals=` and used only in an SG test (`tests/test_lookahead.py:460`). `Selection.pregrasp` is read only by an SG test (`:525,534`) | `candidates.py:133-196` |
| `run_together`, `run_skill`, `SkillExecution` | Essential. `run_skill`'s external branch is the only real-robot driver; it is exercised by the `external` scenario (`test_grasp_place.py:272,393`). G never calls `run_skill` and rebuilds it as `drive` because `run_skill` lacks `tick` | `execution.py:125-202` |

---

## 3. Public-surface classification

Categories: (a) Geodude plus SG behavior tests (no second application exists); (b) required for a stable boundary with one consumer; (c) application-specific, candidate to move to Geodude; (d) compatibility-only; (e) unsupported by demonstrated behavior.

### The 18 exports added since the baseline

| Export | Class | G evidence | Notes |
|---|---|---|---|
| `Candidate` | a | `recycling.py:797-806` | — |
| `Candidates` | a, and Geodude-shaped | `recycling.py:1025,1069` | Do not generalize: arm race plus a drop prefix only. `releases=` is (e) |
| `Selection` | b | structural only: `.candidate`, `.grasp` (`recycling.py:1041-1048,1136`) | Return type; needs to exist. `.pregrasp` is (e) |
| `Criterion` | e | none | Evidence field only; no reader |
| `Observed` | e | none | Evidence field only; no reader |
| `EstimateReason` | b | read as `.value` into an artifact (`recycling.py:596`) | The evaluator does not branch on it |
| `DropSpec` | a | `recycling.py:460-475` | — |
| `drop` | a | `recycling.py:446,1053` | `planned=` is (e) |
| `LandingSpec` | c | `recycling.py:504-507` | Simulation evaluation only; see §4 |
| `landing` | c | `recycling.py:450,1114` | Same as `LandingSpec` |
| `Follow` | **e** | none | Superseded by `Candidates(after=)`; ssgrasping#21 decided not to adopt it |
| `Lookahead` | a | `recycling.py:1043` | Only `unsettled` is used by G |
| `Reason` | a | `recycling.py:859,1122`; `tests/test_recovery.py` | — |
| `Transition` | a | `recycling.py:563,570,861` | Members `ACQUIRED` and `RELEASED` are (e): never estimated |
| `TransitionEstimate` | a | `recycling.py:568-570` | — |
| `TransitionEvaluation` | b | structural (`recycling.py:578-583,724`) | Never named, but its fields are read |
| `TransitionOutcome` | a | `recycling.py:597,724-731` | — |
| `run_together` | a | `recycling.py:349,1026` | G also re-implements it (`_drop_racing_next`) |

### The 10 baseline exports

| Export | Class | Notes |
|---|---|---|
| `GraspSpec` | a | `deepest` and `approach_tries` are (e): G removed `deepest` (geodude 71e31c8) and no longer has `regions.deepest` |
| `PlaceSpec` | a | Worktop path, `recycling.py:405-422` |
| `grasp` | a | `follow=` is (e) |
| `place` | a | — |
| `SkillExecution` | a | `.upcoming` is (e) |
| `SkillResult` | a | `.values` is a dict protocol (§1.6) |
| `SkillState` | a | — |
| `Stage` | a | G constructs it in `tests/test_recovery.py:21` |
| `Evidence` | a | — |
| `run_skill` | b | SG tests and README only; it is the real-robot (external clock) driver |

Non-exported internals imported by consumers:
- `ssgrasping.drop.transfer_request` (`tests/test_lookahead.py:107`): (e), Follow-only.
- `ssgrasping._skill.Run.begin`, monkeypatched by G scripts: an undeclared hook (§1.6).
- `ssgrasping._poses.inside` (`scripts/study_transitions.py:519`).

---

## 4. Recommendations

### Ranked immediate deletions and consolidations

**1. Delete `Follow` and its hand-off chain.**
- **Removed:**
  - `Follow` (`_skill.py:190-200`);
  - `Run.upcoming` and the `close` hand-off with its aliasing trick (`_skill.py:288-290, 381-389`);
  - `Run.ahead(allowances=)` (`_skill.py:310-321`);
  - `SkillExecution.upcoming` (`execution.py:112-116`);
  - `grasp(follow=)` and `hand_on` (`grasp.py:171,191-194,358-364,372`);
  - `Selection.grasp(follow=)` (`candidates.py:121,129`);
  - `drop(planned=)`, `transfer_request` and `_check_no_retreat` (`drop.py:137,150-155,162-163,169-197,259-263`);
  - the export and the README section.
- **Behavior kept:** G's recycling is unchanged; it never uses this path.
- **Proof:** G `tests/test_four_cans.py` and the frozen corpus (no `Follow` import in G). In SG, the `lookahead`, `race` and other scenarios of `tests/test_lookahead.py`, minus `transfer_ahead` and `transfer_disturbed` (`:259-332, 695-716`).
- **Removes:** about 4 concepts (`Follow`, `upcoming`, `planned=`, `transfer_request`), about 110 production lines and about 100 test lines.
- **Knock-on:** ssrobot `predict(allowances=)` (`ssrobot/src/ssrobot/context.py:1276,1371`) loses its only consumer. Flag this to the ssrobot auditor; uncertain whether anything else uses it.
- **Note:** ssgrasping#21 stays open for "a drop with a retreat". Close the Follow half explicitly.

**2. Delete `GraspSpec.deepest` and `approach_tries`.**
- **Removed:**
  - the two fields with their docs and validation (`grasp.py:133-141,156`);
  - `_deepest_approach` (`432-455`);
  - `_distances`' variable branch (`471-485`);
  - the deepest branches in `ahead` and `ahead_request` (`272-284, 292-298`);
  - `values["approach_m"]`;
  - the `deepest` scenario (`tests/test_lookahead.py:553-572,818-819`);
  - mentions in the `candidates.py` docstring (`:13-15`).
- **Behavior kept:** the approach stays at `approach_m`, which is what G runs (`recycling.py:381`).
- **Proof:** G's corpus and SG `test_grasp_place`.
- **Removes:** 1 concept and about 65 production lines. Reintroduce it when a gripper model supports deep grasps *and* an application asks (geodude 71e31c8 parked it).

**3. Route `place._descend` through `ctx.execute(plan, planner=, guard=spec.guard)`.**
- **Removed:** the hand-rolled `prepare`, `GuardedJointTrajectory` and `ctx.submit` (`place.py:349-358`) and their imports (`place.py:58,67`). `_descend` becomes a `Run.move`-like call.
- **Behavior kept:** the guarded descent. It also gains the trace's `compiled` record, which `ctx.execute` writes (`context.py:1943-1947`).
- **Proof:** SG `test_grasp_place` `success` and `no_contact` scenarios, and G `test_recycling` (worktop).
- **Removes:** about 10 lines and one parallel execution path. Uncertain: equivalence of validators. `ctx.prepare` uses the planner's validator, as `_descend` does, but verify byte-identical evidence.

**4. Delete the write-only `values` keys:** `closed_opening`, `lifted_opening`, `hand_allowance`, `contact_pose` and `descent` (`grasp.py:317,350,377`; `place.py:277-280,363-367`).
- **Behavior kept:** all of it. These values are restated in `gripper[...]`, in `Criterion`s and in stage details, or read by nobody.
- **Proof:** no reader in SG tests or scripts, or in G (grep, §1.6).
- **Removes:** about 15 lines and 5 implicit protocol keys. Keep `gripper`, `attach_transform` and `tare_rest`, which tests read, until `values` is typed (larger candidate 2).

**5. Remove `Transition.ACQUIRED` and `Transition.RELEASED`, or make them real estimates.** They are declared (`outcomes.py:81-86`) but never produced.
- **Behavior kept:** all of it.
- **Proof:** grep shows no producers.
- **Removes:** 2 enum members, which also ends the three-way "released" ambiguity. Alternatively, estimate `released` at the release stage and have G read the estimate instead of `s.name == "release"` (`recycling.py:627`).

**6. Remove `Candidates(releases=)`.** It is superseded by `removals=` for G (`candidates.py:143,154`). Proof: G uses `removals` (`recycling.py:1075`); move SG's `race` scenario (`tests/test_lookahead.py:460`) to `removals`. Removes 1 parameter. ssrobot `predict(releases=)` may still have other uses; ssrobot auditor.

**7. Serialize ssrobot records with `dumps` instead of hand-flattening.** That means `rest_evidence` (`_skill.py:242-259`), the prediction fields in `lookahead_entry` (`_skill.py:228-237`) and the report copy (`_skill.py:640-646`).
- **Better still:** log only the prediction's trace id and fingerprint, and the continuation `source`, and let the ssrobot trace be canonical.
- **Proof:** G `test_four_cans` already cross-checks against the trace.
- **Removes:** about 35 lines and one duplicate schema.

**8. Stop deriving the evidence kind from prose** (`_skill.py:654`). Use a typed discrepancy kind from ssrobot, or fix the evidence at `PLANNED` and leave the provenance to the trace. This is 1 line, but it is a protocol defect.

### Ranked larger candidates

1. **One planning-ahead record.**
   - Replace `values["lookahead"]` with a typed `Lookahead` entry on `SkillResult`, or drop it in favour of the ssrobot trace plus a stage→execution-id map. Today G and SG tests assert on its prose (`test_four_cans.py:98,111`; `tests/test_lookahead.py:704,707,722`).
   - Keep only the dynamic boundaries; the static `close`, `restrict` and `release` boundaries are documentation.
   - This would remove about 120 lines of `Run` (`_log`, `boundary`, `adopt`, `decision` bookkeeping) and one untyped protocol shared across two repositories.
2. **Typed, versioned `SkillResult` serialization owned by ssgrasping.**
   - Make the outcomes ssrobot `Record`s, or give them one `to_json` with a schema version, and replace the open `values` dict with typed optional fields.
   - That deletes the four ad hoc flatteners in G (`recycling.py:542-623`) and the SG test `_report` (`test_grasp_place.py:312-350`).
   - It belongs with geodude#246.
3. **One feasibility definition per skill.**
   - Today, `grasp()` alone checks pregrasp→approach (`Run.choose`). Raced, the same skill checks the full prefix through ssrobot chains. The `Run.choose` seed loop (`_skill.py:813-865`) and `Candidates._chain` (`candidates.py:197-298`) are parallel implementations.
   - Option: make `grasp(spec)` a one-candidate `Candidates` internally, and do the same for drop and place transfers, so that ssrobot chains are the only seeded-prefix planner.
   - Risk: plan-seed changes shift later CBiRRT plans. Cross plans before attributing outcome changes.
4. **One driver.**
   - Let `run_together` admit items added during `tick`. It already re-reads `items` each loop (`execution.py:189-201`).
   - Give `run_skill` a `tick`.
   - Then G's `drive`, `_drop_racing_next`, the landing tick and `_wait` (`recycling.py:344-352, 996-1003, 1005-1012, 1189-1222`) collapse to calls. This is the G side of geodude#245.
5. **Make `landing` an evaluation, not a skill.**
   - Once ssgrasping#36 lets a landing wait on observed object rest (`ObjectRest` exists: `ssrobot/src/ssrobot/rest.py:45`), `landing` can become `ctx.watch_rest(ObjectRest)` followed by a pure `score_landing(...) -> TransitionEvaluation`.
   - That removes `Wait.until_ns`, a `Run` and `Stage` for something that commands nothing, and `Reason.STILL_ATTACHED` (its only producer, `landing.py:81`).
   - The evaluation could then attach to the drop's estimate without G's cross-result `(stage, transition)` join.
   - Possibly move it into the application (c), since it is simulation-only.
6. **Split `Run`.** Separate a stage recorder (record, estimates and evaluations) from the planning-ahead machinery (`ahead`, `kept`, `_continue`, `_replanned`, `rest`). This is not fewer lines, but it would let a reader understand grasp→drop without lookahead.
7. **Retire the private `Run.begin` hook.** G scripts monkeypatch it (`calibrate_release.py:138`, `time_startup.py:86`). Use `SkillExecution.stage` polled in `tick`, or a trace record, so that no private symbol is load-bearing.
8. **Take the default hand allowance from ssrobot.** `_hand_frames` (`candidates.py:409-422`) and `Candidates._held` (`:300-316`) reimplement ssrobot's default-allowance rule and attach transform. ssrobot should expose a hypothetical `Attachment` builder; uncertain whether one already exists (ssrobot auditor).

### Do not generalize further (one demonstrated consumer)

- **`Candidates`:** keep it to grasp plus an optional drop. No place prefix, no other skills, no N-skill chains until ADA needs to race.
- **The `Lookahead` knobs** (`arrival_tolerance`, `rest_timeout_ns`): G uses only `unsettled`.
- **The `released` callback:** G's `assume_landed` and `register` (`recycling.py:425,446`). No callback registry or extra hooks.
- **`TransitionEstimate` evidence** (`Observed`, `Criterion`): no new estimate kinds or criteria taxonomy until a second transition uses portable evidence. Today only `retained` does (`_skill.py:536-583`).
- **`landing` and `LandingSpec`:** simulation evaluation only. No generalization to placements (place already scores itself inline, `place.py:315-317`).
- **`Candidates.report` summary semantics** (`canceled_by_priority`): leave them to ssrobot's `ChainsReport`, and do not grow them in ssgrasping.
- **Planning ahead across skills** (`Follow`): delete rather than extend to retreats (ssgrasping#21's open half).

---

## 5. Verdict: can ssgrasping be understood independently of Geodude?

**Partly.** The core contract can be read alone: the specs, the staged generator, `SkillResult` with `Reason`/`cause`, and the estimates. The module docstrings plus the README reasons table (`README.md:~50-80`) explain it without Geodude. `src` names Geodude only in two timing comments (`_skill.py:108,114`).

**What blocks independence:**
1. **One robot.** Every behaviour test and study runs Geodude (`test_grasp_place.py:368`, `test_lookahead.py:228`, `study_transitions.py:299`). The two-robot proof, ssgrasping#6, is open. "Robot-neutral" is asserted, not demonstrated.
2. **Geodude-shaped features.**
   - `Candidates` is an arm race plus a container drop.
   - `removals=` exists for G's registered-pose policy.
   - Parked features are understandable only through Geodude history: `Follow`, which G declined in ssgrasping#21, and `deepest`, which G parked in geodude 71e31c8.
3. **Implicit protocols shared with Geodude:**
   - `values["lookahead"]` prose asserted in G tests;
   - stage-name strings G branches on;
   - the `Candidates.report` dict G's evaluator parses;
   - a private `Run.begin` hook G scripts patch.

   A maintainer cannot change any of these without reading G.
4. **No owned serialization.** Each consumer re-flattens `SkillResult` its own way, so the durable shape of a skill outcome is defined in Geodude (`recycling.py:542-623`), not in ssgrasping.
5. **Mechanism density.** There are six ways to plan a stage, three representations of a plan made ahead (`Kept`, `Ahead`, `Selection._start`), and two feasibility definitions for one skill. A reader has to learn ssrobot's projection, prediction and continuation, plus SG's wrappers, before following one grasp→drop.

Deletions 1–6 remove about 7 concepts and about 200 production lines without touching any path Geodude runs. Larger candidates 1–2 remove the cross-repository dict and prose protocols, which are the main obstacle to reading the layers independently.
