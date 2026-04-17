# Changelog

## v3.0.0 (2026-04-17)

203 commits since v2.0.0. Near-total rewrite of internals; public API stable but behavior and architecture transformed. Coordinated with mj_manipulator v2.0.0.

### New features

- **Natural language chat** — IPython console with integrated LLM (Anthropic Claude) for robot control via natural language. Browser chat panel in Viser with thinking indicator, timestamps, collapsible tool calls, cost tracking. `run_python` tool lets the LLM execute arbitrary robot API. (#96, #139, #142)

- **Browser-based viewer** — `--viser` flag launches mj_viser web viewer with teleop tab, F/T sensor panels, tabbed layout (Chat + Sensors), status HUD, and Stop button. Replaces mjpython for daily use. (#131, #144, #152)

- **Unified arm scope** — `robot.left` / `robot.right` provide a single interface with all arm, gripper, and planning methods. Tab completion works in IPython. Replaces the split `robot.left_arm` / `robot.left_base` API. `robot.left.close()` / `robot.left.open()` for gripper control. (#108, #104, #125, #129)

- **Demo system** — `python -m geodude --demo recycling` with scenario discovery, `--list-demos`, and `save_demo`. IPython console as the primary interface. Removed standalone example scripts. (#101)

- **Generalized placement** — `robot.place()` works on surfaces (boxes, cylinders), not just containers. Derives placement surfaces from destination geometry via face enumeration. Drop clearance adapts to held object size. (#110)

- **LiftBase BT node** — moves the Vention base up after grasp, checks surface clearance as the post-condition (not contact inspection). Writes `grasp_confirmed` to blackboard. (#173, #106)

- **Perception integration** — `SimPerceptionService` wired; `robot.perception.refresh()` before planning cycles. (#175)

- **Wrist F/T sensors** — both arms report 6-axis wrench via mj_manipulator's F/T infrastructure. `tare_ft()` forwarding, auto-clear on reset. (#94)

- **Recycling demo expansion** — pool of 11 object types (cans, spam cans, pop-tarts, sugar boxes, wooden blocks, etc.), 5 random per run. Herring tin removed (too flat for reliable grasping).

### Architecture changes

- **Primitives rewrite** — `pickup` / `place` / `go_home` reduced from 581 → 315 LOC. Delegates to mj_manipulator primitives with Geodude-specific additions (VentionBase homing, geodude_pickup/place BT subtrees). (#170)

- **BT recovery removed** — centralized `_recover` in primitives replaces per-subtree recovery. Simpler, more predictable failure handling. (#109)

- **Fork-based planning** — `_plan_single` uses `env.fork()` for isolated planning instead of teleporting the Vention base. No side effects during plan exploration. (#187)

- **GraspSource delegation** — `PrlAssetsGraspSource` from mj_manipulator replaces Geodude's copy. (#162)

- **Console delegation** — generic console infrastructure from mj_manipulator. (#161)

- **StatusHud delegation** — uses mj_manipulator's `StatusHud`, deleted Geodude's copy. (#168, #169)

- **Scoped teleop** — per-arm ownership via `OwnershipRegistry`; `_arm_preempted` skips unavailable arms. (#159, #164, #165)

### Bug fixes

- **Robotiq 2F-140 grip force** — applied `fix_robotiq_grip_force` to both arms. Kills the menagerie position-actuator bug where grip force → 0 at full close. (#190)

- **E-Stop persistence** — `clear_abort()` no longer called automatically in primitives; user must explicitly release. Per-arm abort separate from global E-Stop. (#186)

- **Object visibility on reset** — move all freejoints to hide position; fix inactive objects not disappearing after recycle in kinematic mode. (#148)

- **Violent motion after reset** — fixed by syncing controller targets. (#d20fa48)

- **Pickup failure detection** — distinguish plan vs grasp failure, use `plan_failure_reason` from blackboard, surface in diagnostics and LLM context. (#116-#121)

- **Chat panel fixes** — Enter-to-send removed (Viser fires per keystroke), Send button only, scrollable history, light theme, markdown rendering. (#139, #140)

- **TSR generation** — fix non-upright objects (shift along local Z, not world Z), fix container drop orientation, fix grasp offset using site frame. (#110)

- **Reset scene** — use demo `spawn_count`, support custom objects, restore fixtures to original positions. (#114)

### Diagnostics

- **Gripper validation** — `scripts/validate_2f140.py` runs mj_manipulator's collision sweep on the geodude_assets 2F-140. (#191)

- **PD gain tuning** — `scripts/tune_ur5e_gains.py` headless harness for the UR5e. (#b8d94cd)

### Breaking changes

- `robot.clear_abort()` no longer called automatically in primitives — user must explicitly release E-Stop.
- `robot.left_arm` / `robot.left_base` removed; use `robot.left` unified scope.
- Geodude BT recovery subtree removed; recovery handled in primitives layer.
- `_find_scene_objects` replaced with `robot.find_objects()`.
- `GenerateDropZone` alias removed; use `GeneratePlaceTSRs`.

### Internal

- 140 tests (was 74 in v2.0.0)
- Open-source prep: MIT relicense, SPDX headers, issue/PR templates (#153)
- `pyproject.toml` version corrected: was `0.1.0`, now `3.0.0`
- `__init__.py` `__version__` corrected: was `0.2.0`, now `3.0.0`

### Known limitations

- Chat panel Enter-to-send doesn't work due to Viser firing `on_update` per keystroke (#140). Use the Send button.
- Browser panels (BT viz, planning diagnostics, scene state, drag-to-place) are scoped for v3.1 (#134-#138).
- HardwareContext for real UR5e + Robotiq is scoped separately (#91).
