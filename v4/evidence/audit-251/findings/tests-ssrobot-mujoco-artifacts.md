# Test audit: ssrobot MuJoCo runtime and artifact boundaries

Scope: personalrobotics/geodude#251, section "Test audit", area "ssrobot MuJoCo and
artifact boundaries". Snapshot: ssrobot `0ecbd8d` (worktree
a worktree at the snapshot), baseline `116f3c8`. Read-only: nothing was edited,
deleted, committed or filed.

## Files taken

`tests/` holds 37 test modules plus `conftest.py`, `support.py` and `fixtures/`
(`franka_panda_parser`, `urdf_arm`). I took these 23 modules, 64 test functions
(65 items with parametrization):

| Area | Modules |
|---|---|
| MuJoCo runtime, model, profile, binding | `test_mujoco.py` (4), `test_mujoco_execution.py` (2), `test_mujoco_gripper_completion.py` (2), `test_mujoco_gripper_opening.py` (1) |
| Observations, rest | `test_mujoco_observation.py` (2), `test_observation_instant.py` (1), `test_scene_observations.py` (1), `test_rest.py` (1) |
| Scenes, scene assets, planning scenes, materialization, rendering and viewers | `test_mujoco_scene.py` (2), `test_mujoco_scene_assets.py` (1), `test_mujoco_planning.py` (3), `test_planning_clearance.py` (1), `test_registered_scene.py` (1, MuJoCo/planning-scene parts only), `test_materialize_world.py` (1), `test_mujoco_render_state.py` (1) |
| Guards in MuJoCo | `test_guarded_execution.py` (2), `test_opening_guard.py` (1) |
| Run bundles, traces, `read_trace` | `test_run_bundle.py` (1). `read_trace` has no dedicated test; it is exercised as the reader in `test_mujoco_scene.py`, `test_mujoco_execution.py`, `test_mujoco_planning.py`, `test_scene_observations.py` |
| Package loading and ingress (MJCF, URDF, SRDF, Menagerie, inference, authoring) | `test_loaders.py` (7), `test_packages.py` (8 functions, 9 items), `test_inference.py` (10), `test_authoring.py` (8) |
| Packaging, installed wheel, core imports | `test_packaging.py` (2) |

I left these to the contracts/planning auditor: `test_chains`, `test_conformance`,
`test_continuation`, `test_contracts`, `test_motion_validation`,
`test_offset_continuation`, `test_plan_alternatives`, `test_planner_portfolio`,
`test_planning_adapter`, `test_planning_contracts`, `test_runtime_contract`,
`test_schemas`, `test_trajectory_contract`. Some tests inside my files are
context/contract logic run through ReplayRuntime or the context. I list them and
defer them, marked **(deferred)**: `test_packages.py::test_overlapping_groups_share_ownership`,
`::test_subgroup_fault_stops_the_composite_everywhere`, the continuation halves of
`test_registered_scene.py` and `test_opening_guard.py`, and the `check_command` rate
cases of `test_packages.py::test_declared_joint_limits_change_the_description_and_its_checks`.

The reference-robots and installed-wheel gates are CI steps, not pytest tests
(`.github/workflows/ci.yml`, jobs `installed` and `reference-robots`). They are named
below where they give stronger coverage.

## Verification performed

- I ran all 23 modules against the snapshot source. I used `PYTHONPATH=<worktree>/src`,
  the existing `.venv` interpreter from a local ssrobot checkout,
  `-p no:cacheprovider`, `--basetemp` and `SSROBOT_ARTIFACTS` in a scratch directory, and
  `--durations=0`. Result: **65 passed in 21.9 s** on macOS. `git status` on the
  worktree stayed clean. Log:
  a scratch log.
- Interval history (`git log 116f3c8..0ecbd8d -- tests/<file>`). Six files were created
  in the interval: `test_mujoco_gripper_completion` (#220), `test_rest` (#223, narrowed
  in #225), `test_planning_clearance` (#183), `test_opening_guard` (#228) and
  `test_registered_scene` (#232). Five were touched: `test_mujoco_execution` (#223),
  `test_mujoco_planning` (#170, #172, #232), `test_materialize_world` (#232) and
  `test_scene_observations` (#232). The rest are unchanged since before the baseline.
- Every module in my area was created in the same commit or PR as the feature it
  covers. Each module docstring states its contract and failure modes before its
  assertions. I found **no module added after its implementation without an
  independently stated failure**. The after-the-fact cases are individual assertions,
  listed under "Trims".

## Headline judgments

1. The area is already end-to-end in style. Each module is one or two scenario tests
   that write an inspectable JSON artifact, and most run twice for byte identity. There
   are no mock-only tests. Runtime cost is small: the slowest test takes 1.9 s and the
   whole area 22 s. The cost to cut is maintenance and duplicated scenario code, not
   CI time.
2. Only four whole-test deletion or merge candidates have solid evidence, plus two with
   low confidence. Most of the accidental complexity is at assertion level: about a
   dozen sub-assertions are tautological, change detectors, or duplicates.
3. The main shared-artifact opportunity is consolidation, not speed: (a) one
   pedestal-grasp scenario re-simulated in 8 modules; (b) about 18 "run it twice and
   compare bytes" checks; (c) two independently built scene-tree run bundles.

## Per-test table

Column key. **Protects**: the externally meaningful behavior, invariant, trust boundary
or failure mode. **Stronger cover**: an existing stronger artifact, if any. **Flags**:
CD = change detector, RI = reimplements production logic, AI = added after the
implementation with no stated failure, X = expensive or shares an artifact.
**Verdict**: K = keep, T = keep but trim named assertions, M = merge or delete
candidate, D = deferred.

### MuJoCo runtime, model, profile, binding

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_mujoco.py:39` `test_mujoco_runtime_resolves_and_steps_the_package` | Every description frame, joint, actuator and allowance binds to the compiled model. Exact integer-ns time with substeps. Velocity commands without velocity actuators are unconfirmed and refused. A failed reopen leaves no stale mapping (`stale_description`, then `not_open`). | Conformance bundle (`ssrobot.mujoco.conformance`, CI `--check`) covers lifecycle on another robot, but not the failed-reopen or mapping-identity properties. | CD: the exact channel list `:117-127` and command list `:111-116` freeze the example manifest's order, not a contract. | T |
| `test_mujoco.py:221` `test_mujoco_runtime_refuses_mismatches_before_commands` | Trust boundary at open: 17 mismatches (stale description, wrong or ambiguous model, MJCF vs URDF disagreement, files edited or symlinked after load, keyframe outside limits, MuJoCo version drift, non-integer timestep) each fail with a stable code and leave nothing open. The start-tolerance edge opens and settles (#100). | None | None | K |
| `test_mujoco.py:438` `test_mujoco_compiles_the_profiles_model_against_the_canonical_description` | URDF-canonical package runs through its profile's MJCF; the keyframe agrees with the SRDF state (#94, #98). | Possibly the CI `reference-robots` job, if Geodude/ADA are URDF-canonical (unverified). | None | K |
| `test_mujoco.py:482` `test_mujoco_model_signature_covers_every_file_the_model_brings_in` | Model signature covers included files and is location-independent (#105). This is the identity that bundles and materialization rely on. | None for the package include. Bundle and materialize tests cover the scene side only. | None | K |
| `test_mujoco_execution.py:287` `test_mujoco_executes_trajectories_chunks_and_grippers` | Trajectory reaches goal. Chunk applies latest due step per tick. Gripper range clipped once to what every finger reaches; reversed maps. Cancel holds last setpoint. Joint resting past a stop can still be commanded (#96). Timeout and `goal_not_reached`. **Trace grows linearly** for sampled trajectories (`:305-313`, real failure #130). | None | X: runs 11 scenarios twice for byte identity (1.8 s) | K |
| `test_mujoco_execution.py:381` `test_mujoco_rejects_unexecutable_commands` | Profile ingress: malformed profile, missing actuator or gripper, ambiguous or unknown profile, actuator alias, empty finger range. Zero gear is unconfirmed with a reason. Overlapping chunk components are refused. | None | Duplicate: the velocity refusal `:395-404` repeats `test_mujoco.py:57-61,134` | T |
| `test_mujoco_gripper_completion.py:215` `test_a_gripper_command_ends_when_its_fingers_stop` | A gripper command ends when the fingers stop: reached vs stopped short, direction, stationary targets, reaching taking precedence. Servos keep the target afterwards. The detector never reads contacts (no privileged control). | None | RI (accepted): `_decided` `:197-212` restates the completion rule as the oracle over recorded readings. It is a verifier of a declared rule against external readings, not a copy of runtime state. Monkeypatches private `runtime._touching` `:183-186` to prove contacts are unread. | K |
| `test_mujoco_gripper_completion.py:260` `test_a_gripper_without_a_completion_detector_is_refused` | Missing detector gives `no_gripper_completion` on MuJoCo and Replay alike. | None | None | K (cheap) |
| `test_mujoco_gripper_opening.py:57` `test_a_profiled_joint_measures_the_opening` | Profile-declared measuring joint gives the calibrated opening. Misreading actuated joints was a real failure. Foreign joint and degenerate calibration are refused at open. | None | None | K |

### Observations and rest

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_mujoco_observation.py:103` `test_mujoco_observes_robot_state_sensors_and_cameras` | Readings stamped `now`. Robot starts at the keyframe. tcp pose in `base` matches an independently compiled model. **Depth at image centre equals `mj_ray` along the camera's optical axis** (#103). Wrist force carries the hand's weight. `arm_qf` balances gravity. | `test_observation_instant.py:172` checks `tcp_pose` every tick within 1e-9, so the single-instant tcp block `:125-140` adds little. The camera, wrench-physics and effort checks are unique. | RI or tautological: the per-reading shape and dtype checks `:116-122` repeat what `RobotContext.observe` already enforces through `check_observation` (`src/ssrobot/validation.py:487`, called from `context.py:658`). | T |
| `test_mujoco_observation.py:169` `test_mujoco_leaves_unbacked_channels_unconfirmed` | A wrench channel with no MuJoCo sensors is unconfirmed (`no_force_torque_sensors`) and observing it is refused. | CI installed step "camera channels are unconfirmed, not fatal" covers the same mechanism for cameras only. | None | K |
| `test_observation_instant.py:172` `test_every_reading_is_of_the_state_after_the_last_step` | Every reading in one observation describes the post-step instant (#162; one-step stale poses were a real failure). Wrench equals the copy's sensors exactly. Observing changes no integration state, warm start included. Channel order does not change effort or opening readings. | None. It is the strongest instant oracle. | FK oracle is `materialize_world_from` (production), not an independent compile, though the base-frame transform is computed in the test. | K |
| `test_scene_observations.py:300` `test_scene_objects_and_contacts_are_observed_and_marked` | Scene pose and contact channels are marked privileged in `RuntimeInfo` and in the trace's `opened` record. `portable_only` refuses privileged channels on MuJoCo and Replay. A runtime cannot smuggle an undeclared robot channel or a robot frame as a scene channel (trust boundary). Contact timing and force are checked against free-fall and `m g`. | "Observing changes nothing" `:96-113,158-164,350` is a weaker version of `test_observation_instant`: it compares snapshot JSON, not `mjSTATE_INTEGRATION`. | X: the `_states` pair runs 2 × 700 ticks, twice. The free-fall checks largely test MuJoCo's integrator, but they serve as the oracle that `ball.pose` is post-step. | T (merge the observing-changes-nothing part) |
| `test_rest.py:438` `test_joints_and_objects_come_to_rest` | Rest watches fire only after settling, not at a velocity zero crossing. Chatter within or beyond the declared bound. Stale or missing channel gives `unavailable`. Object rest on privileged pose. Results not retained by the context (memory). Refusals. Each result is verified against the test's own recorded readings. | None | Not RI: `_justified`/`_within` `:203-250` verify a returned window as a witness. They do not recompute when the watch should fire. Partly context logic; the watch is generic. | K |

### Scenes, scene assets, planning scenes, materialization, viewers

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_mujoco_scene.py:146` `test_grasp_carry_release_and_drop_are_traced` | Box starts where the scene puts it (not keyframe padding). Friction carry. Attachment traced with allowances. Drop is reported as a runtime violation and stays declared until close. Scene revisions and sources in the trace. | None | X: the pedestal-grasp scenario (see "Shared artifacts") | K |
| `test_mujoco_scene.py:248` `test_invalid_attachments_and_scenes_change_nothing` | Refused attach or detach leaves the scene unchanged (12 inputs including bool, float and negative revisions). Uncomposable scenes fail at open. | `missing_include` and `missing_mesh_file` `:258-260` overlap `test_mujoco_scene_assets.py` `unresolvable_reference`. | Attach-validation half is context logic. | K |
| `test_mujoco_scene_assets.py:208` `test_scenes_include_files_and_assets_under_one_identity` | **Trust boundary**: scene references escaping the root by `..`, symlink or absolute path are refused before compile. Identity covers includes and meshes and is location-independent. Declared root. Empty references. **TOCTOU**: the compiled model uses the hashed bytes (`racing`). | None | CD: `reads_of_the_mesh == 1` and the exact `lookup_keys` list `:239-240` freeze MuJoCo-VFS key spelling and read count. `versions_of_the_mesh == 1` plus the `racing` case carry the real property. It also monkeypatches `Path.read_bytes` globally `:162-172`. | T |
| `test_mujoco_planning.py:136` `test_planning_scenes_are_isolated_and_agree` | Planning scenes from one snapshot are mutually isolated and isolated from the live context. FK matches an independent compile at 1e-12. Held box moves with the hand. A decoded snapshot gives identical answers. | CI installed step "planning scene builds on sscbirrt's native scene" checks two of the same probes from the wheel (a different boundary: the installed artifact). | None | K |
| `test_mujoco_planning.py:232` `test_planning_refuses_what_does_not_apply` | Ingress for planning queries: foreign-world snapshot, wrong shape, NaN, string or bool configurations, renamed joint, missing object, unknown allowance, stale and foreign `submit(snapshot=...)`. | None | Partly context logic. | K |
| `test_mujoco_planning.py:281` `test_an_allowed_static_robot_frame_is_honoured` | Regression for a real bug: allowances on robot frames the arm does not move reached sscbirrt only for moved bodies. Allowances swap correctly within one shared planning model. | None | None | K |
| `test_planning_clearance.py:292` `test_planning_keeps_the_requested_clearance` | Clearance is honoured against an independently measured 4 mm vs 6 mm gap. Self-collision is unaffected. Evidence made at 5 mm is refused at 0 mm (validation binding). One compiled model and native scene per group regardless of clearance (#170 performance). Runtime model untouched. | None | CD: the cache counts `:336-348` monkeypatch private `_planning.planning_model` and `_runtime.unchanged` `:161-167`, and freeze `package_checked == 2`. The real property is "compiled once per allowance set". The `validation_mismatch` refusals are contract logic. | T |
| `test_registered_scene.py:392` `test_registered_object_poses_hold_until_registered_again` | MuJoCo/planning parts: planning-scene geometry follows the registration, not simulator truth. Removed objects leave planning. A registration error persists through attach and carry. Violation report survives for evaluation. | None | Uses private `ssrobot.context._compose/_inverse` `:71`. The continuation halves (`_continued`) are **(deferred)**. Created in the interval (#232) with its PR. | K / D |
| `test_materialize_world.py:235` `test_a_recorded_world_materializes_without_a_runtime` | A bundled world rematerializes body poses to 1e-9 without a runtime, without stepping physics. `world_mismatch` for an edited scene or foreign package. Materializations share nothing. Registered-state placement. | None | Tautological: `removed_as_planning_places_it` `:219-220` compares `mapping.apply` with `object_qpos`, and `apply` itself calls `object_qpos` (`src/ssrobot/mujoco/_materialize.py:84`; planning calls it at `_planning.py:234`). It cannot fail unless the call is removed. X: builds its own scene tree, run and bundle twice (shared with `test_run_bundle`). | T |
| `test_mujoco_render_state.py:155` `test_a_viewer_reads_runtime_state_and_cannot_change_it` | Viewer handles are copies: vandalizing one changes neither runtime nor trace nor another viewer (compared by model contents). Handle refuses after reopen (`stale_render`). Replay refuses (`render_unavailable`). | None | `drawn_offscreen` `:132,150` is recorded but never asserted. It may legitimately be false without GL, so it is dead evidence, not a test. | K |

### Guards in MuJoCo

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_guarded_execution.py:286` `test_guarded_descent_stops_on_contact_and_completes_without_it` | Wrench guard ends on exactly the tick the test's own readings predict, then holds. Projection sign. Debounce. No contact means it completes. Cancel carries no outcome. Unevaluable guards are refused before motion. | None | Partial RI: `_predict` `:99-106` reimplements the debounce count but calls production `guard.measure`, so a bug in `measure` would not be caught by the tick oracle. The `toward`/`away` and `transient` cases are independent checks. X: 7 descents twice (1.9 s, slowest in area). | K |
| `test_guarded_execution.py:343` `test_guard_outcomes_must_reproduce_from_their_evidence` | Context validates a runtime-reported guard outcome against its own window (trust boundary: the runtime is untrusted). | None | Isolated validator test (contract area). Uniquely protects the boundary, since an honest runtime never produces these outcomes end to end. | K / D |
| `test_opening_guard.py:228` `test_an_opening_guard_stops_a_carry_when_the_fingers_close_past_the_object` | Opening guard triggers on the predicted tick when the box is pulled out, and completes when held. Wrong-quantity guards refused. Guarded plan and continuation plumbing. Wrench-guarded motions are not predicted. | Same tick-oracle pattern as `test_guarded_execution`, but a different channel and quantity. | Created in the interval (#228) with its feature. Continuation half **(deferred)**. | K |

### Run bundles and traces

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_run_bundle.py:172` `test_a_run_bundles_portably_and_verifiably` | **Trust boundary for a persisted artifact**: missing, extra or modified file; manifest path escape or absolute path; symlink; trace naming another world, each with its own code. Byte-identical rewrite. Moved copy verifies. Failed writes leave nothing. Only referenced trace assets bundled. Opens without importing MuJoCo. | The MuJoCo conformance bundle (`conformance/mujoco/bundle`, CI `--check`) verifies one honest bundle from the installed wheel, not tamper cases. | X: own run and bundle (shared with `test_materialize_world`) | K |

### Package loading and ingress

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_loaders.py:70` `test_menagerie_franka_loads_from_its_scene` | Real upstream MJCF (Menagerie Panda at a pinned commit) parses with class and childclass limits matching published Franka specs. Location-independent report. | None | None | K |
| `test_loaders.py:365` `test_mjcf_constructs` | 39 MJCF constructs: supported ones resolve as MuJoCo would; unsupported or ambiguous ones fail explicitly. Includes are contained (`path_escape`, `include_cycle`, duplicate-by-spelling or symlink). | None | Oracle is hand-written expected values, not MuJoCo itself. Non-blocking: a differential check against `mujoco.MjModel` for the accepted cases would be a stronger, independent oracle. | K |
| `test_loaders.py:473` `test_urdf_with_and_without_srdf` | SRDF yields exactly the semantics a package could declare by hand. URDF alone is kinematics only. SRDF end effector needs a declared TCP. | None | None | K |
| `test_loaders.py:690` `test_urdf_textures_are_resolved_and_hashed` | Texture containment (escape by `..` or symlink, foreign `package://`) and hashing (#63). | None | None | K |
| `test_loaders.py:712` `test_franka_parser_fixture_matches_its_provenance` | Integrity of the vendored oracle fixture: upstream bytes match pinned hashes and mesh placeholders stay empty. It protects the validity of `:70`'s oracle, not product behavior. | None | Fixture guard (cheap). | K |
| `test_loaders.py:740` `test_urdf_and_srdf_ingress` | 27 URDF/SRDF mutations fail at ingress with stable codes. Contradictory SRDF is validated, not trusted (#64). | None | None | K |
| `test_loaders.py:751` `test_mjcf_effort_limits_bound_effort_commands` | MJCF `actuatorfrcrange` becomes a symmetric effort bound that `check_command` enforces (#116). | — | Duplicate: the asymmetric case `:787-791,803` repeats `MJCF_CASES["asymmetric actuatorfrcrange"]` `:136-139`. | T |
| `test_packages.py:84` `test_example_package_loads_identically_wherever_it_lives[minimal_arm, bimanual_lift]` | Example reports are schema-valid. Copy loads identically. Installed module load equals source load without importing. | Location independence: `test_loaders.py:98-99`, `test_packages.py:599`. Installed equality and no-import: `test_packages.py:530-569`. Schema validity of reports: `test_loaders.py:76-77,480`, `test_inference.py:67-73`. | Largely subsumed. | M |
| `test_packages.py:111` `test_semantics_identify_each_part_of_the_robot` | Semantic model distinguishes bimanual-on-lifts parts (#7). | Partly `test_authoring.py:158` (bimanual manipulators, hands, grippers via CLI). | CD-ish: asserts the example manifest's content. Low confidence. | K (or M, low confidence) |
| `test_packages.py:163` `test_overlapping_groups_share_ownership` | Ownership on joints across overlapping groups. | — | Context logic on Replay | D |
| `test_packages.py:232` `test_subgroup_fault_stops_the_composite_everywhere` | Subgroup fault propagation (#56). | — | Context logic on Replay | D |
| `test_packages.py:496` `test_end_effector_attachment_is_validated` | Hand and gripper must sit on their own arm's branch (#59). | None | None | K |
| `test_packages.py:530` `test_installed_discovery_runs_no_package_code` | **Trust boundary**: discovering an installed package imports neither it nor its parents (#57). | None | None | K |
| `test_packages.py:572` `test_package_ingress_rejects_bad_packages` | 36 manifest, path, containment and limit cases at ingress (symlink escapes, FIFO, widened limits, foreign `package://`). | None | None | K |
| `test_packages.py:599` `test_contained_symlinks_load_identically_at_two_locations` | Report does not leak resolved absolute paths for internal symlinks (#58). | Partly `CASES["asset file symlink inside the root"]`, which only proves it loads. | None | K |
| `test_packages.py:617` `test_declared_joint_limits_change_the_description_and_its_checks` | Declared TOML limits enter the description, fingerprint and report (#99). | — | Rate and acceleration cases `:674-705` are `check_command` contract logic. | K / D |
| `test_inference.py:76` `test_unambiguous_robots_get_the_expected_entities` | Load-time `[inference]` adopts arm, gripper and TCP on three robots. Franka gives `no_tcp_candidate`. | The Franka entities are asserted identically through the CLI in `test_authoring.py:133-140`. That is a different public interface (`init` writes explicit semantics; `[inference]` infers at load) over the same candidate engine. | Overlap | K |
| `test_inference.py:139` `test_dual_arm_on_lifts_stays_ambiguous_until_resolved` | Ambiguity is never auto-resolved. Report mode adopts nothing. Overrides resolve. | Resolved entities also asserted via CLI in `test_authoring.py:208-226`. | Overlap | K |
| `test_inference.py:229` `test_bad_overrides_fail_at_ingress` | Override ingress. | None | None | K |
| `test_inference.py:248` `test_inferred_semantics_are_never_commandable_by_themselves` | Inference grants no command capability (safety property). | None | None | K |
| `test_inference.py:295` `test_inference_is_off_unless_requested` | Inference never runs implicitly. | — | **Vacuous second assertion**: `manipulator("arm")` raises on `bimanual_lift` whether or not inference runs, because its manipulators are `left_arm`, `right_arm` and their lifted variants (`test_packages.py:117`). The first assertion only checks a field. | M (rewrite) |
| `test_inference.py:333` `test_candidate_ids_never_collide` | Candidate ids encode names, so overrides address distinct chains (#69). | None | None | K |
| `test_inference.py:391` `test_inference_reconciles_with_declarations` | Inference adds only what is missing (#70). | None | None | K |
| `test_inference.py:518` `test_overrides_on_declared_candidates` | Overrides on declared candidates conflict; on partly declared ones they decide only the manipulator (#70). | None | None | K |
| `test_inference.py:552` `test_reasons_are_literally_true` | Reasons do not falsely claim maximality (#72). | None | CD: substring matching on prose (`"maximal"`, `"derived from"`, `"below '...'"`). Rewording breaks it without a behavior change. | K (non-blocking: assert a structured field if one exists) |
| `test_inference.py:570` `test_tcp_candidates_cover_the_tip_body` | A TCP above the gripper frame (Geodude's Robotiq mount) is a candidate. | None | None | K |
| `test_authoring.py:91` `test_unambiguous_arm_completes_on_defaults` | `ssrobot init/inspect/doctor` as a user runs them. Idempotent rerun. No invented end effector. Nothing commandable by default. Schema-valid reports. | None | None | K |
| `test_authoring.py:158` `test_bimanual_lift_requires_and_records_choices` | `init` stops with every open decision (exit 2, nothing written). Answers produce the declared structure and templates. | None | None | K |
| `test_authoring.py:231` `test_nothing_commandable_without_opting_in` | No templates means no commands or channels. | `test_authoring.py:133-140`: the Franka `init` without templates asserts `commands: 0, channels: 0` through the same CLI. | Subsumed | M |
| `test_authoring.py:241` `test_writes_are_safe` | Atomic manifest writes: refuse a different manifest, failures keep the old one, no temporary files, `--force` replaces. | None | None | K |
| `test_authoring.py:289` `test_prompting_uses_the_same_decisions` | Interactive flow asks exactly the open questions plus templates. | None | `isinstance` asserts `:325-327` restate type annotations (tautological). | T |
| `test_authoring.py:349` `test_doctor_verifies_the_built_wheel` | `doctor --wheel` catches a wheel missing a needed file. | None | None | K |
| `test_authoring.py:374` `test_overwrite_never_destroys_a_valid_manifest` | Invalid manifest rejected before the file is touched, even with overwrite. | `test_authoring.py:263-269,275-277`: the CLI `init --force` with an invalid description keeps the old manifest and leaves no temporary files. The CLI calls the same `write_manifest(..., overwrite=args.force)` (`src/ssrobot/cli.py:107`). | Subsumed | M |
| `test_authoring.py:403` `test_operational_errors_are_diagnostics` | Operational failures (missing or corrupt files, corrupt wheel, duplicate answers) give exit status and message, never a traceback. | None | None | K |

### Packaging and core-import gate

| Test (file:line) | Protects | Stronger cover | Flags | Verdict |
|---|---|---|---|---|
| `test_packaging.py:58` `test_core_imports_no_backend` | Core imports no simulator, planner or learning backend; public API closed. Runs on the source tree. | The CI `installed` job runs the same script on the **built wheel** (`--installed`, plus per-extra `--integration mujoco/planning`) and records import reports. That is strictly stronger, but runs only in CI and only on Python 3.11. | Redundant in CI, but the only local or 3.13 run. Cheap (0.3 s). | K (low value) |
| `test_packaging.py:64` `test_gate_finds_backend_imports_in_every_form` | Detection power of the gate itself (eager, lazy, from, aliased, multiline, dotted, `import_module`) and integration boundaries. A gate with false negatives fails silently. | None | None | K |

## (1) Tests that uniquely protect something: keep

The high-value core. Each is a trust boundary, a persisted-artifact boundary, or a
real invariant no other gate checks:

- **Persisted-artifact trust boundaries**: `test_run_bundle.py:172`,
  `test_mujoco_scene_assets.py:208`, `test_materialize_world.py:235` (`world_mismatch`
  and no-physics materialization), `test_mujoco.py:221` (package changed after load),
  `test_mujoco.py:482`.
- **Package ingress and containment**: `test_packages.py:572,530,599,496`,
  `test_loaders.py:365,740,690,70,473`, `test_inference.py:229,248,333`.
- **Simulator-truth invariants with independent oracles**:
  `test_observation_instant.py:172`, `test_mujoco_observation.py:103` (camera and
  effort parts), `test_mujoco_planning.py:136,232,281`,
  `test_planning_clearance.py:292` (independently measured gaps),
  `test_mujoco_render_state.py:155`.
- **Event-driven completion (no magic durations)**:
  `test_mujoco_gripper_completion.py:215`, `test_rest.py:438`,
  `test_guarded_execution.py:286`, `test_opening_guard.py:228`.
- **Profile binding**: `test_mujoco_execution.py:381`,
  `test_mujoco_gripper_opening.py:57`, `test_mujoco_observation.py:169`, `test_mujoco.py:438`.
- **CLI and gate behavior**: `test_authoring.py:91,158,241,349,403`, `test_packaging.py:64`.

## (2) Deletion and merge candidates

Each candidate names the failure mode considered, the coverage that replaces it, and
any unique boundary it protected.

| # | Candidate | Failure mode considered | Replacing coverage | Unique boundary lost | Confidence |
|---|---|---|---|---|---|
| 1 | `test_authoring.py:231` `test_nothing_commandable_without_opting_in`: delete | `init` emits commands or channels the user never opted into (unsafe default). | `test_authoring.py:133-140` asserts `commands == 0, channels == 0` after a CLI `init` with no templates. | Only that the no-templates rule also holds when answers come from an `--answers` file rather than defaults. The template switch is the same code path either way. | High |
| 2 | `test_authoring.py:374` `test_overwrite_never_destroys_a_valid_manifest`: merge into `test_writes_are_safe` | `write_manifest(overwrite=True)` truncates or replaces a valid manifest before validating the new one. | `test_authoring.py:263-269` (invalid description with `--force` keeps old bytes) and `:275-277` (no temporary files). `cli.py:107` calls the same `write_manifest(..., overwrite=args.force)`. | A direct library-API caller of `write_manifest`. Same function, so it is not a separate boundary. | High |
| 3 | `test_inference.py:295` `test_inference_is_off_unless_requested`: rewrite, do not delete | Inference runs implicitly and adds guessed semantics. | Nothing covers it today, because the second assertion is vacuous (see table). | The property is real and currently unprotected. Replace with: load `tests/fixtures/packages/franka_panda_parser` (no `[inference]`) and assert zero manipulators and grippers, which inference in adopt mode would create (`test_inference.py:95-99`). | High that it is weak. The fix is a recommendation. |
| 4 | `test_packages.py:84` `test_example_package_loads_identically_wherever_it_lives` (2 items): merge | Load result depends on location; installed load imports package code or differs from source; report is not schema-valid. | Location: `test_loaders.py:98-99`, `test_packages.py:599-606`. Installed equality and no-import: `test_packages.py:530-569`. Report schema validity: `test_loaders.py:76-77,480`, `test_inference.py:67-73`. | It is the only test that writes the two example packages' `package-report.json` and `description.json` evidence, and the only installed load of `bimanual_lift`. Move the schema check for the examples into the ingress test's `unchanged` case before deleting. | Medium |
| 5 | Scene-channel "observing changes nothing" in `test_scene_observations.py:96-113,158-164,350`: merge into `test_observation_instant.py:143-151` | Observing a channel mutates simulation state, so observed and unobserved runs diverge. | `test_observation_instant.py` compares the full `mjSTATE_INTEGRATION` (stronger than snapshot JSON) but only with robot channels. Adding `contacts=` and the scene channels to its request covers both. | None after the merge. Before it, scene channels' side-effect freedom is protected only by the weaker snapshot comparison. | Medium-high |
| 6 | `test_packages.py:111` `test_semantics_identify_each_part_of_the_robot`: optional merge | The semantic model cannot represent bimanual-on-lifts unambiguously. | `test_authoring.py:208-226` and `test_inference.py:186-204` assert the same manipulators, hands and grippers. | Disjointness of left and right joints `:122-123` and the single-arm tool case `:127-129` (the tool case is also in `test_inference.py:101-107`). | Low; keep is defensible |

### Trims: delete the named assertions, keep the test

| Location | Problem | Evidence |
|---|---|---|
| `test_mujoco_observation.py:116-122` | Tautological: re-checks shape and dtype that `check_observation` (`validation.py:487`, `context.py:658`) enforces on every `observe`. | A runtime returning the wrong shape would raise inside `ctx.observe` before these lines. |
| `test_mujoco_observation.py:125-140` (tcp block) | Subsumed by `test_observation_instant.py:109-120,177` (every tick within 1e-9). | The keyframe-start assertion `:128` is the only unique line. Keep it. Medium confidence, because the instant test's FK uses `materialize_world_from` rather than an independent compile. |
| `test_mujoco_execution.py:395-404` | Duplicate of `test_mujoco.py:57-61,134` (velocity command gives `unavailable_command`). | Same package, same code path. |
| `test_mujoco.py:111-127` | Change detector: exact order and content of the example's commands and channels. | Asserts manifest declaration order. The semantic checks (frames and joints map 1:1, unconfirmed velocity) are elsewhere in the test. |
| `test_loaders.py:787-791,803` | Duplicate of `MJCF_CASES["asymmetric actuatorfrcrange"]` (`:136-139`). | Same input, same code. |
| `test_materialize_world.py:219-220` | Tautological: `apply` places removed objects by calling `object_qpos` (`_materialize.py:84`). | Keep `removed_independent_of_prior_data` and `removed_below_world_m`. The planning-agreement claim is true by construction. |
| `test_mujoco_scene_assets.py:239-240` (`lookup_keys`, `reads_of_the_mesh`) | Change detector on MuJoCo VFS key spelling and read count. | `versions_of_the_mesh == 1` and the `racing` case (`:175-181,243-244`) carry the TOCTOU property behaviorally. |
| `test_planning_clearance.py:336-348` | Change detector on private call counts (`package_checked == 2`) via monkeypatching `_planning.planning_model` and `ssrobot.mujoco._runtime.unchanged`. | Keep `compiled == 1` and `native_scenes` per group (the #170 performance contract). Drop the exact `package_checked` count, which freezes internal control flow. Uncertain: the author may treat the pre/post package check as a contract. |
| `test_authoring.py:325-327` | `isinstance` checks restate type annotations. | mypy runs in CI. |
| `test_mujoco_render_state.py:132,150` | `drawn_offscreen` is computed and written but never asserted. | Either assert it under `MUJOCO_GL=osmesa` (CI has it) or drop it. As written it is neither test nor oracle. |

## (3) Expensive or shared artifacts to generate once and check by several gates

Runtime is small (22 s for the area), so these are consolidation opportunities. They
are worth doing only if they reduce scenario code. They are not a speed fix.

1. **Pedestal grasp-and-hold.** The sequence move `ABOVE`, open, move `LEVEL`, close,
   attach is re-simulated, often twice, in `test_mujoco_scene.py` (`_grasp`),
   `test_mujoco_planning.py` (`_hold`, used by 3 other modules),
   `test_mujoco_gripper_opening.py`, `test_planning_clearance.py`,
   `test_mujoco_gripper_completion.py` (`close_on_box`), `test_opening_guard.py`
   (`_lift`, twice per run, run twice), `test_registered_scene.py` (3 times, run twice)
   and `test_mujoco_render_state.py`. Three slightly different helper copies exist
   (`test_mujoco_scene._grip`, `test_mujoco_gripper_completion._grip`/`_move`, and
   `_hold`). One session fixture could produce the held snapshot and trace once. The
   planning, clearance, opening and render checks would then read it. Caveat: several
   tests need a live context afterwards, so only the snapshot-consuming checks
   (`test_mujoco_planning.py:136,232`, `test_planning_clearance` `support_allowed`) can
   share a precomputed artifact.
2. **Byte-identity reruns.** About 18 tests in this area run their whole scenario twice
   and compare bytes: execution, scene, observation, planning, render, scene assets,
   materialize, instant, rest, scene observations, guarded, opening guard, clearance,
   registered scene, gripper completion, run bundle (written twice) and loaders (copy).
   This roughly doubles the MuJoCo suite's cost. CI already checks reproducibility more
   strongly, across processes and from the installed wheel against committed bytes, with
   `python -m ssrobot.mujoco.conformance --check conformance`. Recommendation: keep
   in-test reruns only where nondeterminism is a credible feature-specific risk: render
   handles (copied state), rest (gc and weakrefs), guards (tick-exact), planning
   (sscbirrt caches), bundles (write determinism). Elsewhere, rely on the conformance
   check plus one rich rerun. Uncertain: the conformance robot does not exercise
   grippers, guards, rest or scene assets, so dropping a rerun leaves that feature's
   determinism to that one rerun.
3. **Scene-tree run bundle.** `test_run_bundle.py:172` and `test_materialize_world.py:235`
   each build the `_tree` scene (`test_mujoco_scene_assets._tree`), run the arm,
   snapshot periodically and `write_bundle`. The materialize test does this twice. One
   session-scoped bundle (with the dropped cylinder, so snapshots are in flight) could
   feed the tamper matrix, the moved-copy check, the no-MuJoCo open and the
   materialization pose oracle. The committed `conformance/mujoco/bundle` is a third
   honest bundle that the tamper matrix could also use. Caveat: it has no mesh or
   include tree.
4. **Independent oracle model.** An independently compiled `arm.xml` (plus scene) is
   rebuilt in `test_mujoco_observation._independent`,
   `test_mujoco_planning._independent_tcp` and `test_planning_clearance._lid` and
   `_self_collision`. This is cheap; share it only if a common helper reduces code.

## Uncertainties and limitations

- Timings are from one macOS run, not from CI hosts. Relative ordering is what matters.
- I did not check whether the `reference-robots` job covers the URDF-canonical
  MuJoCo-profile path (`test_mujoco.py:438`). I kept the test on that uncertainty.
- Candidate 4 and the tcp-block trim depend on the other auditors agreeing that the
  named replacement tests stay. If `test_loaders` or `test_observation_instant` changes,
  re-check.
- I did not audit `test_schemas.py`. Note for the consolidator:
  `test_schemas.py::test_checked_in_schemas_match_the_records` runs
  `scripts/generate_schemas.py --check`, which the same CI `check` job already runs as
  its own step (`ci.yml`, "generate_schemas.py --check") before `pytest`. That is a
  likely exact duplicate in the same job.
