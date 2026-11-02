# ssrobot production code: flows 1 and 2, concentration points, public surface

Audit for personalrobotics/geodude#251. Read-only.

Snapshot: ssrobot 0ecbd8d, ssgrasping d29a4b4, geodude 0618279 (`v4/`). Baseline ssrobot
116f3c8. Paths below are relative to each worktree under
``.
Unprefixed `src/`, `docs/`, `tests/` paths are ssrobot. "S" = ssgrasping production code
(`ssgrasping/src`), "G" = Geodude 4.0 production code (`geodude/v4/src`, `geodude/v4/scripts`).
"Consumer" means S or G, never ssrobot's own tests or docs.

How usage was measured: word-boundary grep of each name over S, G, their tests, and
ssrobot's tests/scripts/examples/conformance, then manual reading of every `from ssrobot
import` block in S and G (S: `_skill.py:29-54`, `candidates.py:64-81`, `grasp.py:68-81`,
`drop.py:44-54`, `place.py:56-68`, `execution.py:22`, `landing.py:28`, `outcomes.py:26`,
`_poses.py:10`; G: `recycling.py:69-93`, `scene.py:40-41`, `regions.py:19`, scripts). No
consumer imports a submodule such as `ssrobot.chains`. Legacy Geodude (outside `v4/`) uses
none of the 46 names. ssgrasping's history in this clone is 23 commits deep, so "never used"
for ssgrasping history is limited to those commits.

Size context: `src/ssrobot/context.py` 1520 to 2971 lines; `portfolio.py` 445 to 915;
`ssrobot.__all__` 144 to 190 names (`src/ssrobot/__init__.py`).

---

## 1. Flow 1: construct a context, plan, prepare, execute one motion

The canonical journey is `README.md:127-151` and works as written (it is the CI example,
`README.md:155-157`).

**Concepts a caller must understand (12):** `RobotPackage`/`load_package`,
`RobotDescription`, a `Runtime` (`MujocoRuntime`), `RobotContext` as a context manager,
`ManipulatorView` (`ctx.manipulator()`), `SceneSnapshot`, a `Planner` (`CBiRRT`),
`PlanResult`, `Execution`, `ExecutionStatus`/`ExecutionState`, the clock (`run_until` or
`step`), and optionally `JsonlTrace`. `PlanningRequest` and `PreparedTrajectory` are
implicit in this path (built by `plan_to_*`, compiled inside `execute`).

**State transformations and serialization boundaries:**

| Step | Owner | Transformation | Durable form |
| --- | --- | --- | --- |
| `ctx.snapshot()` | `context.py:1013` | live scene to immutable `SceneSnapshot`, registered in `_captured` (`context.py:466`) | `SNAPSHOT` trace record |
| `arm.plan_to_*` / `arm.request` | `context.py:2773-2911` | goal to `PlanningRequest` bound to the snapshot fingerprint (`build_request`) | `PlanningRequest` v3 schema |
| `ctx.plan` | `context.py:1158-1181` | negotiate (`_negotiated_scene`, `context.py:1210-1226`), isolated planning scene, `planner.plan`, `check_result` | `PlanResult` v4 |
| `ctx.prepare` | `context.py:1962-2027` -> `preparation.prepare` (`preparation.py:62-108`) | plan to runtime command, validated; cached in `_prepared` by plan fingerprint | `PreparedTrajectory` v6, `COMPILED` record |
| `ctx.execute` | `context.py:1931-1960` | `prepare` + optional guard + `_submit`; records `_ran` | `Submission` v3, `SUBMITTED`/`STATUS` |
| `ctx.run_until` / `step` / `update` | `context.py:2148-2198` | advance the runtime clock, apply updates | `STEPPED`, `STATUS` |

Flow 1 is coherent: each boundary has one owner and one record. The only duplications in
this flow:

- **Three ways to compile and submit a plan.** `ctx.execute(plan, guard=)`
  (`context.py:1931-1960`), `ctx.prepare` + `ctx.submit`, and the free function
  `ssrobot.prepare` + `ctx.submit`. ssgrasping's descent uses the third
  (`ssgrasping/src/ssgrasping/place.py:349-358`: `prepare(...)`, then
  `GuardedJointTrajectory(...)`, then `ctx.submit`). `ctx.execute` gained `guard=` only in
  #228 (`git log -S"guard: Guard | None = None"`: e27a25a; the baseline signature at
  116f3c8 `context.py:572-579` has no guard). So that ssgrasping path predates the
  replacement and is now superseded. It also skips `_ran` (so the descent can never anchor
  a projection continuation) and the `COMPILED` trace record.
- **Waiting.** `ctx.run_until` (`context.py:2161-2198`) has no consumer: S and G call it
  0 times. Geodude has its own step loop `_wait` (`geodude/v4/src/geodude/recycling.py:817-824`,
  which is `run_until(max_ticks=)` plus a per-tick callback). ssgrasping has its own driver
  `run_together` (`ssgrasping/src/ssgrasping/execution.py:173-205`). The missing per-tick
  hook is a plausible reason that `run_until` goes unused, but I have not confirmed it.

## 2. Flow 2: race alternatives, and continue a plan made ahead

### 2a. Racing

ssrobot offers **three racing front ends on one engine**:

| Front end | Exports | Engine | Consumers |
| --- | --- | --- | --- |
| `plan_portfolio` (pre-baseline) | `PlanAttempt`, `Portfolio`, `PortfolioReport`, `AttemptRecord`, `AttemptOutcome`, `StopReason` | `portfolio._JobRun` (`portfolio.py:406-786`) | **none** in S or G |
| `plan_alternatives` / `start_alternatives` / `ctx.start_alternatives` (#180, #194) | 9 names | `portfolio._JobRun` via private imports (`alternatives.py:38-48`) | **none** in S or G |
| `plan_chains` / `start_chains` / `ctx.start_chains` (#195, #197, #199) | 13 names | `portfolio._JobRun` via private imports (`chains.py:51-61`) | S only: `ctx.start_chains` at `ssgrasping/src/ssgrasping/candidates.py:161` |

Evidence that alternatives have no consumer: 0 hits for any of the 9 names in S, G, S tests,
and G tests. The only references outside `alternatives.py` are `__init__.py`, the lazy
method `context.py:1228-1248`, a stale docstring in `chains.py:5-6,28`, and a comment in
`portfolio.py:342`. `git log -S` finds no use in Geodude history. `README.md:179` says
"Planner portfolios, alternative requests, staged candidates ... used by ssgrasping and
Geodude". **That is false for portfolios and alternatives.**

Concepts a racing caller must hold: `Chain`, `Stage`, `StageRequest`, `StageResult`,
`ChainsRun` (advance/done/planning/cancel/close/result), `Chains`
(`.selected` -> `(Chain, int, tuple[StageResult,...])`, `chains.py:597`), `ChainsReport`,
plus `PlanProjection` (each later stage plans on the previous stage's projection,
`chains.py:76-92`, `805-827`). Each front end has its own report schema (`PortfolioReport`,
`AlternativesReport`, `ChainsReport`). Each also has its own outcome enum mapped from the
one private `_Ended` (`portfolio.py:347-354`; mappings at `portfolio.py:309-315`,
`alternatives.py:94-101`, `chains.py:153-160`), and its own `.selected` tuple shape
(`portfolio.py:250`, `alternatives.py:304`, `chains.py:597`).

Ownership defects in the racing code:
- The shared engine lives in `portfolio.py`, so `chains.py` and `alternatives.py` import
  its privates (`_Job`, `_JobRun`, `_End`, `_Ended`, `_check_fingerprint`).
- `chains.py:5-6` says stages run "on a worker of the alternatives engine
  (`ssrobot.alternatives`)". `chains.py` imports nothing from `alternatives`, so this
  prose is stale.
- The engine writes the context's private caches (`portfolio.py:736-738`:
  `self._ctx._captured`, `self._ctx._prepared`).
- `_state()` is defined twice (`context.py:266`, `chains.py:642`).

### 2b. Planning ahead and continuing

| Step | Owner | Transformation | Durable form |
| --- | --- | --- | --- |
| `ctx.predict(execution, unsettled, releases, removals, allowances, joints)` | `context.py:1269-1428` | executing motion's end + carried objects (`_carried`, `1430-1451`) + declared scene changes, into `PredictedSnapshot`; registered in `_predicted` | `PredictedSnapshot` v3, `PREDICTED` record |
| `ctx.plan_async(request, snapshot=prediction)` | `context.py:1183-1208` | `plan` on a worker, as a `Future` | none |
| `ctx.project(plan, snapshot)` | `context.py:1453-1493` | successful plan's end, into `PlanProjection`; registered in `_projections`/`_projected_states` | **none**: `PlanProjection` is a plain dataclass (`continuation.py:321-336`), not a `Record`, and is never traced |
| `ctx.continuation(plan, prediction=)` or `(plan, projection=, source=)` | `context.py:1495-1559`, `_bind` `1561-1609`, `_check_same_end` `1611-1644` | binds the plan to a prediction. The projection mode calls `predict(source)` internally and checks it equals the projection, or that a continued occurrence ends at the same place (`_continued`) | none (`Continuation` dataclass, `continuation.py:339-346`) |
| `ctx.watch_rest(JointRest(...))` -> `RestWatch` -> `RestResult` | `context.py:663-811`, `rest.py:237-324` | joint channel samples to a rest decision; fingerprint recorded in `_rested` | `RestResult` v2, `RESTED` |
| `ctx.execute_continuation(continuation, rest=, ...)` | `context.py:1646-1894` | checks rest evidence, arrival, scene changes, path constraint; `planner.rebase`; `prepare`; `_submit` | `ContinuationReport` v5, `CONTINUED` |

**Concepts added to flow 1 (about 15):** `PredictedSnapshot`, `PlanProjection`,
`Continuation`, `ContinuationResult`, `ContinuationReport`, `Discrepancy`, `JointRest`,
`RestWatch`, `RestResult`, `RestOutcome`, `RestCriteria` (when the runtime declares none),
`Future` (from `plan_async`), the two binding modes of `continuation`, and the rule that a
plan made on a prediction or projection runs only by `execute_continuation`
(`context.py:1990-2001`: `predicted_plan`, `projected_plan`).

**Hidden state acts as the protocol.** Whether a plan may run, and how, depends on which of
nine private registries its fingerprints appear in: `_captured`, `_predicted`,
`_projections`, `_projected_states`, `_ran`, `_continued`, `_accepted`, `_prepared`,
`_rested` (`context.py:466-491`). Only `_captured` and `_prepared` are ever cleared
(`context.py:2341-2342`). The other seven grow for the life of the session. Each check is a
real safety contract, so this is essential. Still, a maintainer must learn all nine maps to
predict a refusal.

**Three kinds of hypothetical state.** These are `PredictedSnapshot` (a traced Record),
`PlanProjection` (untraced), and `SceneSnapshot.with_attachments` (`scene.py:365-390`, a
chain stage's hypothetical grasp, `chains.py:825-827`). The union
`SceneSnapshot | PredictedSnapshot | PlanProjection` appears in 21 signatures (16 in
`context.py`, 5 in `chains.py`). `predict` and `project` share `_carried` but otherwise
build their states separately, with different stamps and different validity rules.

### 2c. Duplicated mechanisms across both flows

- **Plan:** `ctx.plan`, `ctx.plan_async`, `ManipulatorView.plan` plus five `plan_to_*`,
  `plan_portfolio`, `plan_alternatives`/`start_alternatives`/`ctx.start_alternatives`,
  `plan_chains`/`start_chains`/`ctx.start_chains`. ssgrasping adds its own synchronous
  two-stage seed loop, `Run.choose` (`ssgrasping/src/ssgrasping/_skill.py:813-865`: plan,
  `ctx.prepare`, `ctx.project`, then plan the follow-on, retry on the next seed). It is used
  by `grasp.py:247`, `drop.py:232`, and `place.py:233`. This is a two-stage, one-chain
  `Chains` run done by hand.
  `plan` and `plan_async` repeat the same result check (`context.py:1177-1180` and
  `1203-1206`).
- **Two spellings for each race:** module functions (`alternatives.py:446`,
  `chains.py:1100`) and context methods that wrap them lazily (`context.py:1228-1266`). The
  only consumer uses the method. `plan_chains` (`chains.py:1116-1135`) and `plan_portfolio`
  have no consumer.
- **Wait or advance:** `Execution.done`, `RestWatch.done`, `Future`, and
  `ChainsRun`/`AlternativesRun.advance/planning`, plus `run_until` and three blocking loops
  inside `plan_*`. The "Passive" protocol (`done`/`planning`/`advance`) that ssrobot's run
  handles satisfy is declared in ssgrasping (`execution.py:161-170`), not in ssrobot.
- **Record a failure:** raised `SsrobotError(code)`; `PlanResult.status` plus
  `diagnostics`; `ContinuationReport.refusal` (returned, not raised); `RestOutcome`;
  `StageOutcome`/`AlternativeOutcome`/`AttemptOutcome`. `execute_continuation` both raises
  (`unknown_rest`, `wrong_type`, `unknown_prediction`, `planner_contract`, and any
  `prepare` error other than `execution_unready`; `context.py:1683-1712`, `1847`, `1866-1868`)
  and returns refusals. A caller must handle both.

### 2d. Prose, dicts, and implicit state used as protocol

1. **`Discrepancy.what` is a free-form string that encodes kind, subject, and component**
   (`continuation.py:222`; values `"revision"`, `"attachments"`, `"fixtures"`,
   `f"object {name}"`, `f"object {name} rotation"`, `f"object {name} state"` built in
   `context.py:329-418`). Booleans are encoded as `predicted=0.0, actual=1.0`.
   ssgrasping branches on it: `any(d.what.startswith("object ") ...)`
   (`ssgrasping/src/ssgrasping/_skill.py:654`).
2. **`Planner.rebase` is not part of the `Planner` protocol** (`planning_contracts.py:884-910`
   declares `name`, `capabilities`, `plan`, `validator`). It is discovered with
   `getattr(planner, "rebase", None)` (`context.py:1793`), but the docstring cites
   `Planner.rebase` (`context.py:1676`). The rebase account is assembled by matching
   diagnostic codes `"rebase"`/`"rebase_failed"` (`context.py:1849`). This conflicts with
   ssrobot's own rule "Capabilities are declared and negotiated up front"
   (`docs/architecture.md:230-231`).
3. **`ContinuationReport.refusal` is a `str`** checked against a tuple `REFUSALS`
   (`continuation.py:48-59`), not an enum. ssgrasping keeps a subset as a separate tuple
   (`RECOVER`, `_skill.py:116`).
4. **Typed records consumed as JSON dicts.** ssgrasping does `json.loads(dumps(result.report))`
   on `ChainsReport` (`candidates.py:320`) and on `GripperResult` (`_skill.py:725`). It
   then branches on dict keys (`grasp.py:321,379`: `closed["termination"] ==
   GripperTermination.REACHED_TARGET`). It also re-derives "outranked" from the dict
   (`candidates.py:374-376`), duplicating `ChainsReport._validate` (`chains.py:544-546`).
   ssrobot exposes no typed "why this chain has no choice" answer, so the consumer
   recomputes it.

---

## 3. Concentration points

**context.py (2971 lines).** Approximate regions:

| Region | Lines | Size |
| --- | --- | --- |
| helpers, `_scene_changes`, `_ContinuedOccurrence` | 241-420 | about 180 |
| rest watches | 663-811 | about 150 |
| attach/detach/register/remove/snapshot (registered objects) | 812-1086 | about 275 |
| plan, plan_async, start_alternatives, start_chains | 1158-1266 | about 110 |
| predict, project | 1269-1494 | about 225 |
| continuation, binding, execute_continuation | 1495-1908 | about 415 |
| submit, execute, prepare, _submit | 1911-2118 | about 210 |
| views | 2740-2971 | about 230 |

The interleaving section (1269-1908, about 640 lines) is a separate state machine that
already has its own records module (`continuation.py`) yet lives in `RobotContext`.

- **alternatives vs chains: superseded in practice.** Chains is the only racing front end
  with a consumer. A set of one-stage chains expresses "same task for either arm"; ssrobot's
  own chains test already races two arms (`tests/test_chains.py:225`). Alternatives offer
  three things chains lack. (i) A two-level priority (alternative, then attempt). (ii)
  Parallel attempts selected by shortest duration, where chain seeds run in order, first
  feasible wins. (iii) `cancel_lower` scoped to the alternative's priority. No consumer
  relies on any of these.
- **plan vs plan_async.** Both are needed: `plan_async` has one consumer (`_skill.py:330`).
  The bodies repeat about 5 lines.
- **project vs predict.** Not redundant. `project` is pre-submission and stampless, used by
  chains and `Run.choose`. `predict` is post-submission, timed, and accepts declared scene
  changes. Projection-mode `continuation` reduces to `predict` plus an equality check
  (`_bind`, `context.py:1593-1609`). The redundancy is in the representations: an untraced
  `PlanProjection` next to a traced `PredictedSnapshot`.
- **execute vs execute_continuation.** Both are essential and both are used by S
  (`_skill.py:506,599` and `_skill.py:635`). They share the `prepare` + `_submit` tail.
- **Registered objects (#232: `ObjectPoses`, `ObjectStatus`, `Registration`,
  `AttachmentReference`).** Two object-pose regimes, LIVE and REGISTERED, branch through
  snapshot, predict, `with_attachments`, and execute_continuation (15 references in
  `context.py`). G uses only REGISTERED (`recycling.py:261`). LIVE remains the default
  (`context.py:444`) and is used by ssrobot tests and ssgrasping tests
  (`ssgrasping/tests/test_grasp_place.py:378`). `AttachmentReference` is written to the
  trace (`context.py:889-900`, `TraceKind.ATTACHED`, `trace.py:45-47`) "for evaluation",
  but no S or G code reads `ATTACHED` records.
- **preparation.** Clear ownership. The shortfall is the extra public free `prepare` path
  described in section 1.

---

## 4. Public-surface classification: the 46 names added since 116f3c8

Classes: (a) used by at least 2 independent consumers; (b) required for a stable boundary
with one consumer; (c) application-specific, candidate to move down; (d) compatibility-only,
candidate for removal; (e) unsupported by demonstrated behavior (no consumer).

Note: `docs/contracts.md:44-48` makes the interface closed. Every type exposed as a field of
a public name must itself be public. Several (b) entries below are public only because of
this rule (record parts), not because a consumer names them.

| Name | S (src) | G (src/scripts) | Class | Evidence / note |
| --- | --- | --- | --- | --- |
| Alternative | - | - | e | only ssrobot tests; superseded by chains for the one racing consumer |
| AlternativeAttempt | - | - | e | report part |
| AlternativeOutcome | - | - | e | report part |
| AlternativeRecord | - | - | e | report part |
| Alternatives | - | - | e | |
| AlternativesReport | - | - | e | schema `schemas/ssrobot.AlternativesReport.v1.json` (235 lines) |
| AlternativesRun | - | - | e | |
| plan_alternatives | - | - | e | |
| start_alternatives | - | - | e | plus `ctx.start_alternatives` (`context.py:1228`) |
| AttachmentReference | - | - | e | emitted to trace (`context.py:894`), read by no consumer; claimed "for evaluation" (`scene.py:224-227`) |
| Chain | import (`candidates.py:66`) | - | b | single consumer; G sets `priority` through S (`recycling.py:803`) |
| Stage | import (`candidates.py:76`) | - | b | uses `seeds`, `retry_prefix=False` (`candidates.py:268-272`) |
| StageRequest | import | - | b | uses `attachments` (`candidates.py:235-236`) |
| StageResult | import | - | b | |
| ChainsRun | import (`candidates.py:67`) | - | b | |
| Chains | (via `ChainsRun.result`) | - | b | not named |
| ChainsReport | (via JSON, `candidates.py:320`) | - | b | |
| ChainAttempt, ChainRecord, StageAttempt, StageOutcome | - | - | b (closure) | report parts; read only as JSON keys |
| start_chains (module fn) | - | - | d | the consumer uses `ctx.start_chains`; two spellings of one entry |
| plan_chains | - | - | e | blocking wrapper; ssrobot tests only |
| PredictedSnapshot | import (4 files) | - | b | |
| PlanProjection | import (3 files) | - | b | |
| Continuation | import (`_skill.py:31`) | - | b | |
| ContinuationResult | (returned) | - | b | |
| ContinuationReport | (fields read, `_skill.py:638-660`) | - | b | |
| Discrepancy | (`.what` parsed, `_skill.py:654`) | - | b | the protocol is defective; see 2d.1 |
| JointRest | import | - | b | |
| RestWatch | import | - | b | |
| RestResult | import | - | b | |
| RestOutcome | import (`place.py:62`) | - | b | |
| RestSample, RestChannel | (fields read, `_skill.py:246-255`) | - | b (closure) | |
| RestCriteria | - | - | b | runtime-implementer boundary (`ChannelRest`); caller-supplied criteria are unused by S (`_skill.py:668`) |
| ChannelRest | - | - | b | `RuntimeInfo.rest` (`runtime.py:86`); one implementer, ssrobot's MuJoCo runtime (`mujoco/_runtime.py:871-877`) |
| ObjectRest | - | - | e | ssrobot tests only (`tests/test_rest.py:326`); its intended consumer, ssgrasping#36, is parked (`ssgrasping/src/ssgrasping/landing.py:12,48-49`) |
| OpeningGuard | import (`_skill.py:38`) | - | b | |
| GripperResult | (via JSON, `_skill.py:722-725`) | - | b | |
| GripperTermination | import (`grasp.py:70`) | import (`recycling.py:75`) | **a** | the only name of the 46 imported directly by both consumers |
| GripperDirection, GripperEvidence | - | - | b (closure) | fields of `GripperResult`; no consumer reads them |
| ObjectPoses | docstrings; S tests | import (`recycling.py:77,261`) | a/b | production use by G only; S tests exercise both modes |
| ObjectStatus | - | import (`recycling.py:78`) | b | |
| Registration | - | import (`recycling.py:82`) | b | |

Summary: (a) 1 to 2 names; (b) 29 names, about 10 of them public only through the
closure rule; (d) 1; (e) 15. **None are (c).** Nothing here is Geodude-specific enough to
move down. The single-consumer (b) names fail ssrobot's own gate: "No new core abstraction
without two concrete consumers" and "No new top-level export merely for possible future use"
(`docs/architecture.md:224-225,241-242`). For the racing and continuation names, the
honest status is "one consumer, ssgrasping".

Documentation drift: the public-surface table (`docs/contracts.md:50-58`) omits
`ObjectPoses`, `ObjectStatus`, `Registration`, and `AttachmentReference`, all added in #232.
It also lists `BASE_CONSTRAINTS`, which is not in `ssrobot.__all__` (it is defined at
`planning_contracts.py:69`).

---

## 5. Recommendations

### Ranked immediate deletions and consolidations

1. **Delete alternatives.**
   - Deleted: `src/ssrobot/alternatives.py` (520), `ctx.start_alternatives` and its
     TYPE_CHECKING import (`context.py:130,1228-1248`, about 22), 18 lines of
     `__init__.py`, `schemas/ssrobot.AlternativesReport.v1.json` (235),
     `docs/planning.md:384-459` (about 76) and the 9 names in `docs/contracts.md:56`,
     `tests/test_plan_alternatives.py` (881).
   - Behavior that remains: racing distinct requests as one-stage chains, proved by
     `tests/test_chains.py:225` (two arms, deterministic selection).
   - Must port first: three engine contracts currently proved through alternatives.
     These are worker-thread validation (`test_plan_alternatives.py:461`, half of which
     already uses chains at `:525-548`), prepare-before-select (`:759`), and
     passive-while-moving (`:612`). In-flight work is already covered for chains
     (`test_chains.py:473`).
   - Estimate: 9 exports, 1 context method, 1 schema, 1 selection rule, and 1 outcome enum
     removed. About 870 non-test lines, plus about 880 test lines, minus about 150 lines of
     ported tests.
2. **Replace `Discrepancy.what` string parsing with typed fields.** For example, an enum
   `kind` (joint, object, revision, attachments, fixtures) plus a `subject` field. Then
   ssgrasping's `startswith("object ")` (`_skill.py:654`) becomes a field test. This needs
   `ContinuationReport` v6. It adds about 10 lines and removes one implicit protocol. The
   behavior is proved by `tests/test_continuation.py:698`
   (`test_continuations_check_what_they_reconcile`).
3. **Declare `rebase` in the planner contract.** Either add it to the `Planner` protocol or
   negotiate it as a declared capability, and drop `getattr` (`context.py:1793`). Fix the
   `Planner.rebase` docstring reference (`context.py:1676`). This removes one undeclared
   duck-typed method. Proved by the `rebase_unsupported` and `rebase_failed` cases in
   `tests/test_continuation.py:698` and `tests/test_offset_continuation.py:300`.
4. **Retire ssgrasping's hand-rolled descent compile** (`ssgrasping/src/ssgrasping/place.py:349-358`)
   in favor of `ctx.execute(plan, planner=planner, guard=spec.guard)`. This change is in
   ssgrasping. Afterwards the top-level free `ssrobot.prepare` export has no consumer and
   becomes an (e) candidate. About 10 lines in ssgrasping; 1 export later. Caveat: the
   descent's trace would gain a `COMPILED` record. Any Geodude evidence that digests whole
   traces must be checked; I did not verify whether any does.
5. **One entry point per race.** Keep `ctx.start_chains` (the consumer's spelling). Make
   `start_chains` and `plan_chains` non-exported test helpers, or drop the context method
   and keep the module function. 2 exports, about 25 lines, and one of the two documented
   spellings (`docs/planning.md:465-466`).
6. **De-duplicate `plan`/`plan_async`** (`context.py:1177-1180` vs `1203-1206`). Have `plan`
   call a shared `_planned()` helper. About 5 lines. Proved by any planning test plus
   ssgrasping's lookahead tests.
7. **Fix the documentation claims.** `README.md:179` claims use by ssgrasping and Geodude;
   `docs/contracts.md:50-58` has the table drift; `chains.py:5-6,28` has the stale
   "alternatives engine" prose. Fixing these removes no concept, but it stops readers from
   learning concepts that are not live.

### Larger candidates (needs a decision or more proof)

A. **Collapse the racing engine to one front end.** `plan_portfolio` predates the arc but
   has no S or G consumer either. A portfolio is one alternative, which is a set of
   one-stage chains, except for `timeout_s` (`portfolio.py:262-270`), which no consumer
   uses. With both alternatives and portfolio gone, `_JobRun` moves into `chains.py`. That
   ends the private cross-module imports and the context-private writes
   (`portfolio.py:736-738`) and leaves one outcome enum, one report schema, and one
   `.selected` shape. Estimate: `portfolio.py` front end (about 340), `PortfolioReport`
   schema, `tests/test_planner_portfolio.py` (517), and 7 exports. Uncertain: portfolios
   are a documented pre-arc user feature (`README.md:82`), so this is a public-surface
   decision, not cleanup.
B. **Move the interleaving state machine out of `RobotContext`.** `predict`, `project`,
   `continuation`, `_bind`, `_check_same_end`, `execute_continuation`, and five of the nine
   registries (about 640 lines, `context.py:1269-1908`) would move into
   `continuation.py`, behind a small context-internal interface. This relocates code; it
   removes nothing. It makes `RobotContext` readable as "session plus commands".
C. **One hypothetical-state representation.** `PredictedSnapshot`, `PlanProjection`, and
   `with_attachments` states could become one traced record with a provenance field. At
   minimum, make `PlanProjection` a `Record` and trace it, so a continuation bound through
   a projection is reconstructible from the trace. Uncertain: whether the merged record
   simplifies the `continuation` binding rules or only renames them.
D. **LIVE vs REGISTERED object poses.** Decide whether LIVE stays a supported control mode
   or becomes a test and evaluation mode. Geodude 4.0 states that registration is the only
   way object poses enter control (`recycling.py:9`). Removing the LIVE branch from
   predict, snapshot, and `with_attachments` would cut a two-regime cross product, but it
   would affect ssrobot and ssgrasping tests. This is a product decision, not cleanup.
E. **`ObjectRest` and pose rest judging.** Keep these only if ssgrasping#36 (observed
   landing) is unparked. Otherwise remove the `ObjectRest` record and schema (105 lines),
   the pose branch of `watch_rest` (`context.py:711-718,737-738`), the pose path of
   `_Judge`, `POSE_ELEMENTS`, and `RestCriteria.translation_m`/`rotation_rad`.
F. **`AttachmentReference` and `TraceKind.ATTACHED`.** These were added in the snapshot
   commit #232, with no reader. Delete them (record about 25 lines, schema 128 lines, emit
   about 12 lines, plus a `TraceRecord` schema bump), unless geodude#248 evaluation is about
   to read them. Confirm with the owner.
G. **Waiting.** Either give `run_until` the per-tick hook that both consumers needed, so
   Geodude's `_wait` (`recycling.py:817-824`) can go, or stop presenting `run_until` as the
   way to wait. Uncertain why Geodude wrote `_wait` (see section 1).

### Do not generalize further (one demonstrated consumer)

- **Chains** (consumer: `ssgrasping/src/ssgrasping/candidates.py`). In particular
  `start_chains(..., abort=)` and `cancel_lower=False`, which the consumer never passes
  (`candidates.py:161`). Do not add more stage policies beyond `seeds` and `retry_prefix`
  without a second consumer.
- **`predict`'s declared-change parameters** `releases`, `removals`, `allowances`, `joints`,
  `unsettled` (`context.py:1269-1306`). All are driven by ssgrasping alone
  (`candidates.py:153-155`, `_skill.py:320-322`).
- **`execute_continuation`'s tolerance knobs** `object_tolerance_m`/`object_tolerance_rad`.
  The consumer passes only `rest`, `arrival_tolerance`, and `guard` (`_skill.py:635-636`).
- **`OpeningGuard`** (`_skill.py:176` only), **registered objects** (Geodude only), and
  **caller-supplied `RestCriteria`** (no consumer passes one).
- **`ssgrasping` `Run.choose`** is a second sequential staged-planning mechanism next to
  Chains. Before generalizing either one, decide which owns "plan, then check the next
  stage on its projection" (cross-reference for the ssgrasping auditor).

---

## 6. Verdict: can a user or maintainer understand ssrobot independently?

**Flow 1: yes.** A user can understand it from `README.md:127-157` and `docs/planning.md`
"Ordinary planning". It needs about 12 concepts, each boundary has one owner, and each
transformation is a traced, schema-versioned record.

**Flow 2: only with ssgrasping open beside it.** Four things block independent
understanding:

1. **The surface advertises mechanisms with no consumer.** These are portfolios,
   alternatives, `plan_chains`, `ObjectRest`, and `AttachmentReference`, and
   `README.md:179` says otherwise. A reader cannot tell the live path (`ctx.start_chains`,
   `predict`, `plan_async`, `project`, `continuation`, `watch_rest(JointRest)`,
   `execute_continuation`) from the exhibits.
2. **Hidden registries and undeclared protocols decide validity.** Nine session registries
   (`context.py:466-491`) and the undeclared `rebase` method (`context.py:1793`) decide
   whether a plan may run. A string field (`Discrepancy.what`) carries meaning that the
   consumer has to parse.
3. **`context.py` concentrates three subsystems.** Session/commands, registered-object
   scene management, and the about 640-line interleaving state machine all live in one
   2971-line class. The racing engine's ownership is split across `portfolio.py`,
   `alternatives.py`, and `chains.py`, with stale prose about which owns what.
4. **The meaning of a race's outcome is only partly in ssrobot.** ssgrasping re-derives
   "canceled by priority" from the JSON report (`candidates.py:362-395`), so the semantics
   of a chain without a choice are split across two repositories.

The essential parts are sound: snapshot identity, prediction/projection binding, rest
evidence, and prepare-before-execute each protect a demonstrated contract (the frozen
corpus). The accidental parts are concentrated and removable. Recommendations 1-3, 5, and 6
lie wholly within ssrobot and are independent of each other; recommendation 4 is an
ssgrasping change.
