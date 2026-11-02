# Test audit: ssrobot contracts, planning and execution (geodude#251)

Snapshot: ssrobot `0ecbd8d`. Interval baseline: `116f3c8`. Read-only audit. Nothing was changed or deleted.

## Scope

`tests/` has 37 `test_*.py` files plus `support.py` and `conftest.py`. I reviewed **64 test functions, 65 collected items** (`test_records_cannot_shadow_the_wire_envelope` runs twice, once for each of its two parameters).

| Area | Files taken (test count) |
|---|---|
| Records, wire, schemas | `test_contracts.py` (4), `test_schemas.py` (3) |
| Context lifecycle, ownership, runtime trust boundary | `test_runtime_contract.py` (7), `test_trajectory_contract.py` (1), `test_conformance.py` (3, ReplayRuntime and conformance logic), `test_packages.py`: only `test_overlapping_groups_share_ownership` and `test_subgroup_fault_stops_the_composite_everywhere` (2) |
| Planning contracts, adapter, preparation | `test_planning_contracts.py` (5), `test_planning_adapter.py` (7), `test_motion_validation.py` (3), `test_planning_clearance.py` (1, contract part) |
| Portfolios, alternatives, chains | `test_planner_portfolio.py` (4), `test_plan_alternatives.py` (7), `test_chains.py` (3) |
| Prediction, projection, continuation | `test_continuation.py` (5), `test_offset_continuation.py` (2) |
| Guards, grippers, rest, registered state | `test_guarded_execution.py` (2), `test_opening_guard.py` (1), `test_mujoco_gripper_completion.py` (2), `test_rest.py` (1), `test_registered_scene.py` (1) |

The MuJoCo/artifact auditor covers the rest: `test_authoring`, `test_inference`, `test_loaders`, `test_materialize_world`, `test_mujoco`, `test_mujoco_execution`, `test_mujoco_gripper_opening`, `test_mujoco_observation`, `test_mujoco_planning`, `test_mujoco_render_state`, `test_mujoco_scene`, `test_mujoco_scene_assets`, `test_observation_instant`, `test_packaging`, `test_run_bundle`, `test_scene_observations`, and the other 7 tests in `test_packages`.

Some files belong to both areas:
- `test_mujoco_gripper_completion.py`: I took it because the brief lists grippers. The other auditor may also review it.
- `test_planning_clearance.py`: the planning-model cache counts in that test are MuJoCo-internal.

## Evidence I gathered

- **I ran the whole area.** I used the worktree's `src` and the main checkout's `.venv`, which has the same `uv.lock`. Bytecode, the pytest cache and artifacts all went to a scratch directory:
  - Command: `PYTHONPATH=<worktree>/src SSROBOT_ARTIFACTS=<scratch>/contracts-run/artifacts python -m pytest -p no:cacheprovider --durations=0 <files>`.
  - Result: **65 passed in 35.1 s**, EXIT=0.
  - Log: `contracts-run/pytest.log` (a scratch directory).
  - Slowest: portfolio 5.3 s, 4.3 s and 2.7 s; offset continuation 3.3 s; everything else under 2 s. **Cost is not a major problem in this area.**
- **Interval history** (`git log 116f3c8..0ecbd8d`):
  - 26 of the 64 tests were added in the interval. 8 of the files are new: chains, alternatives, continuation, offset_continuation, opening_guard, rest, registered_scene, gripper_completion. New functions also landed in the portfolio, planning_contracts and clearance files.
  - Every added test landed in the same PR as its implementation.
  - Most docstrings state a failure mode with an issue or review reference (#185, #189's review, #202's review, #207, #211). Where none is stated, the table says so.
- **CI gates** (`.github/workflows/ci.yml`):
  - `generate_schemas.py --check` (line 32), then `pytest` (line 33).
  - From the installed wheel: `python -m ssrobot.conformance --out` (line 103) and `python -m ssrobot.mujoco.conformance --check conformance/` (line 106).
- **Downstream e2e** (ssgrasping `d29a4b4`, geodude `0618279`):
  - `ssgrasping/src/ssgrasping/_skill.py`, `candidates.py`, `place.py` and `grasp.py` call `start_chains`, `plan_async` on a prediction, `continuation`/`execute_continuation`, `project`, `prepare`, `watch_rest`, `OpeningGuard`, `WrenchGuard`/`GuardedJointTrajectory`, `clearance_m`, and registered object poses.
  - `ssgrasping/tests/test_lookahead.py`, `test_grasp_place.py` and `test_drop.py`, and Geodude's recycling corpus (geodude#254), exercise these on the happy path.
  - **None of that gates an ssrobot PR, and none of it exercises the refusal paths.** So downstream coverage alone justifies no ssrobot deletion. (Uncertain: I did not read the ssgrasping assertions line by line.)

## Per-test table

Column key:
- **E2E**: whether a stronger end-to-end artifact already covers this test.
- **CD / reimpl / post-hoc**:
  - CD: change-detector.
  - reimpl: the assertion reimplements production logic.
  - post-hoc: added with the implementation and protects no independently stated failure.
- **Cost / shared artifact**: measured call time, and whether the test regenerates an artifact another test already produces.

### Records, wire, schemas

| Test | Protects | E2E | CD / reimpl / post-hoc | Cost / shared artifact |
|---|---|---|---|---|
| `test_contracts.py:218` `test_records_round_trip_through_json_and_checked_in_schemas` | Wire boundary: every record kind validates against its checked-in schema, decodes to an equal value, and re-encodes to the same bytes. Assets are stored once by content. | Partly: conformance traces validate against the TraceRecord schema (`test_conformance.py:109`), but not every record kind. **Unique.** | No | 0.01 s |
| `test_contracts.py:249` `test_joint_command_schema_fixes_the_unit_of_values_by_mode` | Public schema metadata: `x-unit` per mode matches the convention table in docs/contracts.md. | None | The oracle is the documented table, not the code. Low value but semantic. | ~0 |
| `test_contracts.py:300` `test_conventions_accept_valid_and_reject_ambiguous_input` | Ingress validation (about 55 cases): names, frames, quaternions, clocks, limits, documented rate allowances, the START_TOLERANCE stop band, NaN, duplicate keys, asset path escape and integrity, unknown fields, versions. **The core trust-boundary table.** | None | One case, AppliedCommand v1 at `:663-667`, pins a past version number. The generic version case at `:702-705` already covers the mechanism. | ~0 |
| `test_contracts.py:745` `test_contexts_share_a_description_but_not_state` | Context lifecycle: two contexts on one description stay independent; a cross-clock chunk is refused before the runtime sees it; observe-only capability errors; gripper opening outside [0, 1] refused at ingress; a misbound runtime is stale and closed on a failed open; an exception closes the runtime. | `:757-780` overlaps conformance `no_progress_before_step` and `cancel` (`src/ssrobot/conformance.py:439`, `:501`). The rest is unique. | No | ~0 |
| `test_schemas.py:16` `test_checked_in_schemas_match_the_records` | Published schemas do not drift from the code. | **Identical to CI step `ci.yml:32`** (the same `generate_schemas.py --check`). | No, but a duplicated gate. | 0.31 s |
| `test_schemas.py:28` `test_checked_in_schemas_are_valid_json_schema` | Generated schemas satisfy the Draft 2020-12 meta-schema. `validate()` elsewhere does not check the schema itself. | None | No | 0.41 s |
| `test_schemas.py:37` `test_records_cannot_shadow_the_wire_envelope` (×2) | Generator guard: a field named `schema` or `version` would silently overwrite the envelope. | None (isolated by necessity) | No | ~0 |

### Context lifecycle, ownership, runtime trust boundary

| Test | Protects | E2E | CD / reimpl / post-hoc | Cost / shared artifact |
|---|---|---|---|---|
| `test_runtime_contract.py:68` `test_direct_responses_keep_causal_time` | Trust boundary: a runtime answering out of causal order faults the context; deadlines start at acceptance; a manual answer from the future is rolled back. | None. Conformance runs only honest runtimes. | No | ~0 |
| `test_runtime_contract.py:124` `test_applied_commands_stay_within_ownership` | An applied command outside its execution or out of limits is a breach. Nothing misleading reaches the trace. | None | No | ~0 |
| `test_runtime_contract.py:167` `test_invalid_submit_answer_is_rolled_back` | A bad submit answer cancels exactly the handed execution and commits nothing. | None | No | ~0 |
| `test_runtime_contract.py:204` `test_run_until_rejects_a_handle_from_another_context` | `run_until`'s short-circuit for finished executions rejects a handle from another context. | Conformance `terminal_short_circuit` (`conformance.py:579`) covers the short-circuit but not foreign handles. **Unique.** | No | ~0 |
| `test_runtime_contract.py:239` `test_read_trace_enforces_whole_trace_invariants` | Trace ingress: sequence, clock, time, nested stamps, applied ↔ submission linkage; a truncated trace is accepted. | Conformance re-reads only valid traces. **Unique.** | No | 0.01 s |
| `test_runtime_contract.py:333` `test_scene_runtimes_that_break_their_contract` | Attach answers that change the transform or allow set, violations for unattached objects, late answers, missing attach capability. | None | No | 0.01 s |
| `test_runtime_contract.py:499` `test_snapshot_runtimes_that_break_their_contract` | Snapshot and planning-scene answers checked for revision, objects, stamp, group, contact names; a clearance silently ignored is a breach. | None | No | ~0 |
| `test_trajectory_contract.py:106` `test_runtimes_declare_the_trajectory_contract_they_execute` | Each runtime declares the contract it executes, or none: an uneven replay and an external clock declare none. The context answers what the trace kept. | **Partly duplicated.** "ticks == period" is the body of conformance `_check_trajectory_contract` (`conformance.py:418-428`). The mujoco and replay "passed" outcomes are also asserted in `test_conformance.py:89-101`. Unique: the undeclared cases and answered == declared. | Mild CD: `:185` freezes the uneven tick list, which is the input itself. | 0.21 s |
| `test_conformance.py:137` `test_every_runtime_passes_the_conformance_scenario` | **The area's main e2e.** Every runtime passes the checks its capabilities allow. Artifacts are reproducible, traces coherent (an independent oracle re-reads them for transitions and provenance), and the committed artifacts verify while tampered copies do not. | This is the strongest artifact. CI also runs `--check conformance/` from the installed wheel (`ci.yml:106`), so the `conformance_check(committed)` step at `:177` repeats that gate from source. | The `LEGAL` table is restated from docs: a spec oracle, fine. | 0.77 s. `check()` regenerates each runtime twice per directory, and `:177-190` calls it on 4 directories. |
| `test_conformance.py:222` `test_replay_starts_a_trajectory_from_a_joint_resting_past_its_stop` | Runtime clamps a start inside the stop band and reports the modification (#96). | None. The conventions table checks only `check_command` acceptance. | No | ~0 |
| `test_conformance.py:258` `test_replay_faults_on_divergence_and_exhaustion` | ReplayRuntime faults on divergence (recoverable) and exhaustion (not recoverable). | None | No | ~0 |
| `test_packages.py:163` `test_overlapping_groups_share_ownership` | Ownership is by joint: composite and subgroup conflict; partial and shared ownership is reported by `owners()`. | Conformance `ownership` (`conformance.py:474`) covers one group only. **Unique.** | No | ~0 |
| `test_packages.py:232` `test_subgroup_fault_stops_the_composite_everywhere` | A subgroup fault fails the composite in the context and the runtime; recovery stays clean (#56). | **Nearly identical to conformance `composite_fault`** (`conformance.py:768-799`), which passes on replay and MuJoCo in `test_conformance.py:137`. Unique remainder: the right arm stays active. | No | 0.01 s |

### Planning contracts, adapter, preparation

| Test | Protects | E2E | CD / reimpl / post-hoc | Cost / shared artifact |
|---|---|---|---|---|
| `test_planning_contracts.py:489` `test_planning_calls_requests_results_and_execution` | Every planning call yields a fully resolved request (frame, constraints, scaled limits, description untouched). A result carries its request and executes with nothing rebuilt. Unsupported planners are refused. | The adapter carry (`test_planning_adapter.py:178`) is the real-planner e2e. This one uses a scripted planner, so its request-shape assertions are unique. | `_on_polyline` (`:404`) is copied verbatim at `test_planning_adapter.py:457`, a duplicated helper. | 0.18 s |
| `test_planning_contracts.py:526` `test_requests_are_the_same_on_every_runtime` | Invariant: a request depends only on the description and the snapshot, on replay with a decoded snapshot. | None | No | 0.03 s |
| `test_planning_contracts.py:570` `test_malformed_ambiguous_and_falsely_ready_records_are_refused` | Ingress for planning records: ambiguity, scaling, broken or discontinuous ramps, inconsistent results, a falsely ready or rebound PreparedTrajectory. | None | No | 0.06 s |
| `test_planning_contracts.py:759` `test_derived_values_are_computed_once_and_never_shared` | Memoization keyed by object never leaks a fingerprint to a new object at a reused address, never leaks memory, and a request checked for one snapshot is refused for another (#189). | None | Uses private `_wire._MEMO`, an implementation coupling, but the failure (a wrong fingerprint after `id()` reuse) is real. Added in the interval, failure stated. | 0.11 s |
| `test_planning_contracts.py:814` `test_derived_values_are_computed_once_by_concurrent_workers` | Concurrency: compute once, no deadlock on nested memo, a failure is retried. | None. A race is not reachable economically end to end. | Uses private `memo`. Failure stated (#189 review). | 0.06 s |
| `test_planning_adapter.py:178` `test_sscbirrt_plans_a_carry_that_executes` | sscbirrt adapter e2e: the timed trajectory is sampled with no second timing pass, prepared, checked by one validator, and executed while held. A stale snapshot is refused after a scene change. | Downstream skills use the adapter heavily (non-gating). **The gating e2e for the adapter.** | No | 0.88 s |
| `test_planning_adapter.py:295` `test_execute_refuses_requests_this_context_did_not_build` | Trust boundary: a forged request inside a PlanResult (robot, description, revision, start, goal) is refused before submit; a mismatched validator is refused. | None | No | 0.12 s |
| `test_planning_adapter.py:365` `test_neutral_providers_and_geometric_paths` | Python-backend (neutral) provider checked the same way; a geometric-only result needs an explicit processor. | None | No | 0.42 s |
| `test_planning_adapter.py:389` `test_adapter_refusals` | Capability refusals before planning; infeasible and cancelled statuses; a success claim on a colliding path is refused. | `:442-445` (the witness is a collision naming the pedestal) duplicates `:587-591`. | No | 0.11 s |
| `test_planning_adapter.py:478` `test_rest_to_rest_timing_is_exact_and_within_limits` | Mathematical invariant of `RestToRestProcessor`: at rest at every waypoint, on the polyline, within effective limits at 1/2/10/50 ms sampling. | Used in e2e (`:365`), but only at 2 ms. **Unique** for the coarse periods. | `_on_polyline` restates geometry; it is an independent oracle, not production logic. | 0.12 s |
| `test_planning_adapter.py:539` `test_one_validator_answers_everywhere` | One validator answers for state, motion and path; an interior collision is named; repeated queries give equal answers; a path constraint without kinematics has no validator. | None | No | 0.03 s |
| `test_planning_adapter.py:598` `test_evidence_is_bound_not_relabelled` | sscbirrt reports relabelled as another subject or policy are refused (`planner_contract`). `geometry_preserved` is true only when the command traces the path. | None | No | 0.05 s |
| `test_motion_validation.py:376` `test_motion_validator_contract_is_planner_neutral` | Claim: the MotionValidator contract can be implemented without sscbirrt. | Production behavior is covered by `:436`, `:488` and the adapter tests. | **Mostly tests the test's own `IntervalWorld`** (`:128-280`, an exact quadratic-interval validator). The verdict table `:388-410` checks that helper, not ssrobot. Not in the interval. | ~0 |
| `test_motion_validation.py:436` `test_malformed_validation_reports_are_refused` | Ingress: a MotionValidation that contradicts itself does not decode. | None | No | ~0 |
| `test_motion_validation.py:488` `test_reports_are_bound_to_their_exact_subject_and_validator` | Binding: identity covers q(t) and limits, not provenance. Reports about another motion, validator, kind or derived field are refused. Subjects are never coerced. | None | No | ~0 |
| `test_planning_clearance.py:292` `test_planning_keeps_the_requested_clearance` | A requested clearance is kept (lid at 4 mm vs 6 mm, measured independently), including self-collision. Evidence at 5 mm is refused for 0 mm. The planning model is compiled once per group, whatever the clearance. | ssgrasping passes `clearance_m` downstream (non-gating). | The **cache block (`:336-348`) is a partial CD**: it monkeypatches private `_planning.planning_model` and `_runtime.unchanged` and freezes `package_checked: 2`, an incidental count. `compiled: 1` and `native_scenes` per group are the real invariants. | 0.50 s |

### Portfolios, alternatives, chains

| Test | Protects | E2E | CD / reimpl / post-hoc | Cost / shared artifact |
|---|---|---|---|---|
| `test_planner_portfolio.py:358` `test_a_portfolio_selects_deterministically_from_isolated_attempts` | Isolation (a corrupting attempt cannot affect the others or the live context); deterministic selection independent of worker count; the selected candidate executes exactly; expected failures stay with their attempt. | None in ssrobot. Downstream does not use `plan_portfolio`. | No | **5.28 s** (runs `_portfolio` twice) |
| `test_planner_portfolio.py:397` `test_a_timeout_aborts_running_attempts_and_cancels_pending_ones` | Timeout aborts the running attempt and cancels pending ones; a shared planner across workers is refused. | None | No | 0.24 s |
| `test_planner_portfolio.py:414` `test_validating_on_workers_shares_processors_and_honors_a_stop` | Concurrency: a non-reentrant shared processor is never called concurrently; a stop during validation selects nothing (#202 review). | None | Failure stated. | **4.33 s** (sleeps of 0.2 s, 0.5 s and 2.0 s) |
| `test_planner_portfolio.py:480` `test_a_report_cannot_claim_an_impossible_outcome` | Decode refuses impossible PortfolioReports. | None | **Weak oracle:** `:517` asserts only `!= "accepted"`, so a forgery refused for an incidental reason (for example, "reordered" failing a shape check) still passes. Alternatives and chains pin the codes. | **2.73 s, regenerates `_portfolio`**, which `:358` already wrote |
| `test_plan_alternatives.py:267` `test_alternatives_race_two_arms_and_select_deterministically` | Deterministic selection (shorter motion, priority fallback, cutoff, abort); report independent of workers; the candidate executes. | None in ssrobot. ssgrasping races arms through chains, not alternatives. | No | 0.34 s |
| `test_plan_alternatives.py:345` `test_an_alternatives_report_cannot_claim_an_impossible_outcome` | Decode refuses impossible AlternativesReports. | None | No | ~0. **Regenerates `cutoff_1_worker`**, which `:267` writes. |
| `test_plan_alternatives.py:401` `test_alternatives_hold_their_own_inputs_and_classify_outranked_attempts` | Caller mutation after planning changes nothing; an outranked attempt that raises is `canceled_by_priority`; a defect still propagates (#184). | None | Failure stated. | 0.66 s |
| `test_plan_alternatives.py:461` `test_attempts_are_validated_on_workers_while_the_robot_moves` | Validation on workers while the context steps; the passive report equals the blocking one; a shared planner is serialized; a stop during validation is recorded as canceled (#202). | Overlaps `:612` (the passive vs blocking same-report part). | Failure stated. | 1.97 s |
| `test_plan_alternatives.py:612` `test_alternatives_plan_passively_while_the_robot_moves` | Passive run equals blocking while stepping (#191); cancel and close are idempotent; no worker threads are left. | **Same-report-while-stepping is repeated by `:461`.** Unique: cancel/close idempotence and no leaked threads. | No | 0.11 s |
| `test_plan_alternatives.py:691` `test_a_passive_run_has_work_in_flight_until_it_is_done` | Scheduling liveness (#211): a stepping driver never sees an unfinished run with nothing in flight. | None | Failure stated. | 0.04 s |
| `test_plan_alternatives.py:759` `test_a_candidate_whose_command_is_refused_is_never_selected` | Preparation boundary (#207): a candidate valid as planned but refused as compiled is never selected; prior evidence is reused only when verified; `ctx.prepare` refuses a raw plan. | None | Reads private `ctx._prepared` and `ctx._captured` (coupling). Failure stated. | 0.08 s |
| `test_chains.py:225` `test_chains_race_staged_prefixes_and_select_deterministically` | Chain semantics: seeds, stage seeds, prefix retry, disconnected stages, cost accounting, determinism across workers. | `ssgrasping/candidates.py:161` uses `start_chains`, and `test_lookahead.py` exercises it on the happy path (non-gating). The retry and refusal semantics are unique here. | No | 1.57 s |
| `test_chains.py:351` `test_a_chains_report_cannot_claim_an_impossible_outcome` | Decode refuses 14 kinds of impossible ChainsReports. | None | No | 0.41 s. **Regenerates 4 scenarios** (eq, ss, sx, pn) already written by `:225`. |
| `test_chains.py:473` `test_a_chains_run_has_work_in_flight_until_it_is_done` | Race (#211): a stage finishing mid-advance is taken before the advance returns. | None | Monkeypatches private `run._react`, `run._run._pool` and `portfolio._check`. This forces a deterministic interleaving and protects a real race, but is brittle. | 0.04 s |

### Prediction, projection, continuation

| Test | Protects | E2E | CD / reimpl / post-hoc | Cost / shared artifact |
|---|---|---|---|---|
| `test_continuation.py:524` `test_a_plan_made_on_a_prediction_continues_on_arrival` | Plan on a prediction, continue on arrival. Each refusal submits nothing: source not succeeded, not at rest, out of tolerance, unpredicted scene change (attached or falling), rebase unsupported. Exercises unsettled objects, releases and allowance changes. | ssgrasping `_skill.py:330`/`:635` and `test_lookahead.py` cover the happy path (non-gating). **The refusals are unique.** | No | 1.15 s (14 sessions, run twice) |
| `test_continuation.py:698` `test_continuations_check_what_they_reconcile` | Untrusted inputs: bad tolerances, forged or missing rest, continuous-joint wrap at the seam, an object rotated in place, an unheld object, a late start, a blocked bridge, a stale rest window (#185). | None | Failure stated. | 0.33 s |
| `test_continuation.py:912` `test_a_plan_made_on_a_projection_continues_that_execution` | Projection binding (#190): only an execution of exactly the projected plan binds; twins, replans and tampering are refused. | `ssgrasping/_skill.py:856` uses `project` (non-gating). Binding refusals are unique. | No | 0.14 s |
| `test_continuation.py:1058` `test_a_plan_on_a_projection_continues_a_continued_occurrence` | A projection continues a rebased occurrence modulo a turn; a rebase that moves the endpoint is refused (#200). | None | No | 0.40 s |
| `test_continuation.py:1103` `test_the_manipulator_view_plans_on_a_projection` | A manipulator view accepts a Projection as its snapshot. | Indirect: `_projected` in `:912` plans on a projection through `ctx.plan`, not the view. ssgrasping plans through views (uncertain whether on projections). | **Post-hoc API smoke test; no failure stated** beyond "it works". | 0.04 s |
| `test_offset_continuation.py:300` `test_a_continued_offset_motion_keeps_its_planned_line` | An offset motion stays on its planned line through a continuation: inside, outside, bridge, explicit reference, a slid carriage, a misreported arrival (#203). | None | **`old_versions` sub-block (`:357-367`, helper `:270-297`) is a CD.** It hard-codes the previous version numbers from #203 and asserts they decode as `version_mismatch`. The generic mechanism is already in `test_contracts.py:702-705`, and a future missed bump would not be caught. | **3.31 s** |
| `test_offset_continuation.py:435` `test_an_offset_line_is_planned_with_a_declared_margin` | Numerical invariant (#208): a path planned on a narrowed line stays valid under 10/24 µm shifts; without a margin it does not. Bad margins are refused. | None | The diagnostic message string is frozen at `:449-452` (minor CD). | 0.58 s |

### Guards, grippers, rest, registered state

| Test | Protects | E2E | CD / reimpl / post-hoc | Cost / shared artifact |
|---|---|---|---|---|
| `test_guarded_execution.py:286` `test_guarded_descent_stops_on_contact_and_completes_without_it` | The guard ends on exactly the predicted tick, holds, debounces, projects the wrench, completes without contact, and refuses unevaluable guards. | ssgrasping `place.py:357` uses WrenchGuard (non-gating). The exact-tick oracle is unique. | `_predict` (`:96-103`) restates the guard's count rule over independently read wrenches, using production `guard.measure`. It is a spec oracle, not a copy of the runtime loop. | 1.94 s |
| `test_guarded_execution.py:343` `test_guard_outcomes_must_reproduce_from_their_evidence` | Trust boundary: a runtime's GuardOutcome that contradicts its window, threshold, measure, baseline or part is refused. | None. The e2e produces only honest outcomes. | No | ~0 |
| `test_opening_guard.py:228` `test_an_opening_guard_stops_a_carry_when_the_fingers_close_past_the_object` | An opening guard triggers on the predicted tick when the object leaves the hand and completes while held. A guarded source is not predicted; guard and channel kinds must match (#227). | `ssgrasping/_skill.py:176` retention guard (non-gating). | Same pattern of spec oracle on independently observed openings. Failure stated. | 0.64 s |
| `test_mujoco_gripper_completion.py:215` `test_a_gripper_command_ends_when_its_fingers_stop` | A gripper command ends by the declared rule: reached target, stopped short, or stationary; reaching takes precedence (#219). | Downstream grasps exercise it (non-gating). | `_decided` (`:195-212`) restates the completion rule from the declared profile values over observed openings: a spec oracle. | 0.81 s |
| `test_mujoco_gripper_completion.py:260` `test_a_gripper_without_a_completion_detector_is_refused` | Refusal (`no_gripper_completion`) on MuJoCo and replay. | None | No | 0.03 s |
| `test_rest.py:438` `test_joints_and_objects_come_to_rest` | Rest watch semantics: a turnaround is not rest; chatter within and beyond tolerance; stale and missing channels end `unavailable`; object rest; results are not retained; refusals. The certificate is checked against recorded readings. | ssgrasping `_skill.py:667` (non-gating). | The `justified` check verifies the reported window against recorded readings rather than recomputing the outcome. That is a certificate check, acceptable. Failure stated. | 1.21 s |
| `test_registered_scene.py:392` `test_registered_object_poses_hold_until_registered_again` | Registered poses drive planning geometry, not the simulator. Unknown, removed, carried and detached states; continuation refusals; registration refused on live contexts; inconsistent scenes refused (#230). | Geodude #254 runs the recycling corpus on registered poses (stronger, but non-gating and happy path only). The refusal paths are unique. | No | 1.28 s |

## (1) Tests that uniquely protect something: keep (59)

These guard trust boundaries, invariants or races that no stronger gating artifact covers.

- **Records and wire:**
  - `test_contracts.py:218`, `:249`, `:300`, `:745`.
  - `test_schemas.py:28`, `:37`.
- **Runtime trust boundary and ReplayRuntime:**
  - `test_runtime_contract.py:68`, `:124`, `:167`, `:204`, `:239`, `:333`, `:499`.
  - `test_conformance.py:137`, `:222`, `:258`.
  - `test_trajectory_contract.py:106`: its duplicated part is listed in (2).
  - `test_packages.py:163`.
- **Planning contracts and adapter:**
  - `test_planning_contracts.py`: all 5.
  - `test_planning_adapter.py`: all 7. `:389` has a small overlap with `:539`.
  - `test_motion_validation.py:436`, `:488`.
  - `test_planning_clearance.py:292`: its cache counts are listed in (2).
- **Portfolios, alternatives, chains:**
  - `test_planner_portfolio.py`: all 4. The weak oracle at `:480` is a strengthening note, not a deletion.
  - `test_plan_alternatives.py:267`, `:345`, `:401`, `:461`, `:691`, `:759`.
  - `test_chains.py`: all 3.
- **Continuation:**
  - `test_continuation.py:524`, `:698`, `:912`, `:1058`.
  - `test_offset_continuation.py:300` (minus `old_versions`), `:435`.
- **Guards, grippers, rest, registered state:**
  - `test_guarded_execution.py`: both.
  - `test_opening_guard.py:228`.
  - `test_mujoco_gripper_completion.py`: both.
  - `test_rest.py:438`.
  - `test_registered_scene.py:392`.

Keep these isolated tests, despite their coupling to implementation details:
- `test_planning_contracts.py:759` and `:814`: memo and concurrency.
- `test_chains.py:473` and `test_plan_alternatives.py:691`: scheduling races (#211).
- `test_planner_portfolio.py:414`: a shared processor.

They are the only economical way to expose those races and leaks. The private hooks they use (`_MEMO`, `memo`, `run._react`, `run._run._pool`, `portfolio._check`, `ctx._prepared`, `ctx._captured`) are a maintenance cost, not a reason to delete them.

## (2) Deletion and merge candidates (5 whole-test: C1, C2, C3, C5, C8; 5 partial trims of kept tests: C4, C6, C7, C9, C10)

Each candidate names the failure mode considered, the coverage that replaces it, and any unique boundary it protected. None should be deleted in the audit change.

| # | Candidate | Failure mode considered | Replacing coverage | Unique boundary lost / how to keep it | Confidence |
|---|---|---|---|---|---|
| C1 | **Duplicated gate:** `test_schemas.py:16` or CI step `ci.yml:32`; keep one | Checked-in schemas drift from the records | The other one runs the identical command `scripts/generate_schemas.py --check`. Both run in the same CI job. | None. Keeping the pytest test preserves the signal from a local `pytest` run. Dropping it means a local run misses drift until CI. | High |
| C2 | **Delete** `test_packages.py:232` `test_subgroup_fault_stops_the_composite_everywhere` | A subgroup fault leaves the composite running in the context or the runtime, or recovery leaves it applying | Conformance `composite_fault` (`src/ssrobot/conformance.py:768-799`) asserts composite FAILED, recovery OPEN, and 3 clean steps (a runtime still applying would breach the contract). `test_conformance.py:137` requires it `passed` on replay and MuJoCo. | "The other arm stays ACTIVE" (`:264`, `:267`) is not in conformance. Keep it by adding that expectation to `_check_composite_fault` first. **This is a production-code change** (the conformance scenario ships). | Medium-high |
| C3 | **Delete the verdict table of** `test_motion_validation.py:376`, keeping at most a round-trip | That a validator other than sscbirrt cannot satisfy the MotionValidation contract | `:488` already builds an `IntervalWorld` report and passes `check_validation` for path and trajectory subjects. It asserts a continuous policy (resolution None). `:436` covers decode refusal. The adapter tests (`test_planning_adapter.py:539`, `:598`) cover the real validator. | The `:388-410` verdicts and the clip fraction test the test's own `IntervalWorld`, not ssrobot. Only one thing is unique: a continuous-guarantee report round-trips (`:385-386`). Fold one `loads(dumps(report))` into `:488`. | Medium |
| C4 | **Delete the `old_versions` sub-block** of `test_offset_continuation.py:300` (`:357-367`; helper `:270-297`) | A wire-shape change without a version bump | Not caught by this test either: it hard-codes the previous version numbers, so a future missed bump still passes. Strict version refusal as a mechanism is covered by `test_contracts.py:702-705`. | None. The bump policy is a review rule (schema-version policy), which this test cannot enforce. `test_contracts.py:663-667` (AppliedCommand v1) is the same pattern; drop that one row too. | Medium-high |
| C5 | **Merge** `test_plan_alternatives.py:612` into `:461` | A passive run diverging from the blocking run, or not letting the context step | Both run blocking plus passive alternatives on `colliding_bimanual`, with a moving left arm, and assert the same report and stepping during the race (`:507-513` vs `:646-648`). | `:612`'s cancel/close idempotence and no leaked `ssrobot-portfolio` threads (`:649-673`) are unique. Move them into the merged test. | Medium |
| C6 | **Trim** `test_trajectory_contract.py:106`: drop the duplicated mujoco/replay assertions (`:124-126`), or fold the test into `test_conformance.py` | A runtime stepping by something other than its declared period | `_check_trajectory_contract` (`conformance.py:418-428`) performs exactly this check, and `test_conformance.py:89-101` requires it `passed` (mujoco, replay) and `not_applicable` (replay_external). | Unique and must stay: the uneven replay declares none; `ctx.trajectory_contract` raises `trajectory_contract_undeclared`; answered == trace-declared. | Medium |
| C7 | **Trim** `test_planning_adapter.py:389` lines `:442-445` (the colliding-path witness) | Validator witness missing or misattributed | `test_planning_adapter.py:539` asserts the same witness properties (`:587-591`) on a similar motion through the pedestal. | None. `:389`'s refusal codes stay. | Medium |
| C8 | **Merge** `test_continuation.py:1103` into `:912` (plan one leg via `ctx.manipulator().plan_*` on the projection) | A manipulator view rejecting a Projection, or building the request from the wrong state | No direct coverage today. `:912` plans on projections via `ctx.plan`. | The view-level path. Keep it by using the view in `_projected`. Post-hoc smoke test; cheap, so low priority. | Low-medium |
| C9 | **Loosen, don't delete:** the cache counts in `test_planning_clearance.py:336-348` | The planning model is recompiled per clearance (cost), or a native scene per clearance | No other test counts this. | Keep `compiled == 1` and `native_scenes` per group. Drop the frozen `package_checked: 2`, an incidental internal call count from a monkeypatched private function. Belongs to the MuJoCo auditor too. | Medium |
| C10 | **Trim** `test_contracts.py:745` lines `:757-780` | Submit advancing time, cancel not holding the arm | Conformance `no_progress_before_step` and `cancel` (`conformance.py:439`, `:501`), required `passed` on replay and MuJoCo | Two contexts on one description staying independent (`:774-775`) is unique; keep that and drop the rest. | Low |

**Not candidates, despite appearances:**
- **Downstream happy-path coverage:** ssgrasping lookahead/grasp/place and the Geodude corpus do not replace any test here. They are non-gating for ssrobot and never reach the refusal codes these tests pin.
- **Spec oracles:** `_predict` (guarded), `_decided` (gripper completion) and the rest certificate check restate a documented rule over independently observed data. That is acceptable, not a production reimplementation.

## (3) Expensive artifacts that could be generated once and checked by several gates

All measured cost is small (35 s for the area), so these are about duplication more than wall time.

1. **Portfolio** (largest saving, about 2.7 s):
   - `_portfolio(…, workers=1)` runs in `test_planner_portfolio.py:361` and again in `:481` only to obtain `portfolio.json` for forgery.
   - Fix: a module-scoped fixture producing `portfolio.json` once with workers=1. `:358` compares it with workers=3; `:480` reads it.
2. **Chains:**
   - `test_chains.py:351` regenerates 4 of the 8 scenarios that `:225` produces twice (eq, ss, sx, pn).
   - Fix: a module fixture holding the first `evidence()` dict can feed both. `:225` still regenerates once for byte-identity.
3. **Alternatives:**
   - `test_plan_alternatives.py:345` regenerates `cutoff_1_worker` from `:267`. The same fixture pattern applies.
   - C5 also removes one blocking-plus-passive race.
4. **Conformance:**
   - `test_conformance.py:137` runs each runtime twice through `--out`. It then calls `conformance_check` on the committed directory and on 3 tampered copies (`:177-190`). Each call regenerates every runtime twice (`src/ssrobot/conformance.py:938-940`).
   - CI then repeats the committed check from the installed wheel (`ci.yml:106`).
   - The tampered-copy cases only need `_verify` and outcome comparison against one regeneration. Today `check()` offers no way to pass in an existing regeneration, so sharing it needs a small production API change.
   - Measured at 0.77 s, so low priority (uncertain why it is this fast; I did not profile it).
5. **Schema drift:** see C1. The same command runs twice per CI job.
6. **Determinism double runs:**
   - About 20 tests in this area regenerate their scenario twice to assert byte-identity: contracts-planning, adapter, portfolio, alternatives, chains, continuation (×3), offset (×2), guarded, opening, rest, registered, gripper completion, clearance, motion validation, trajectory contract.
   - A single "regenerate every evidence file twice and diff" gate could replace the per-test pattern.
   - At the measured cost this is optional and **not recommended** on cost grounds alone.

## Limitations

- I did not read every helper body in full. For the scripted planners, `_scenario` builders and IntervalWorld internals, I read the definitions and the assertions.
- Downstream coverage claims come from call sites and docstrings, not from verifying ssgrasping and geodude assertions.
- Durations come from one run on this macOS host, using the main checkout's venv with an identical `uv.lock`. They are not comparable with CI.
- "Added after the implementation" means landed in the same PR as the feature. I judged whether a failure was independently stated from the test docstrings and the issue references they cite, not from reading the issues.
