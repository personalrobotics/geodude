"""Natural language CLI for Geodude robot control.

Chat-based interface where an LLM translates natural language to robot
API calls. Works in kinematic, physics, and hardware modes.

Example::

    robot = Geodude(objects={"can": 4, "recycle_bin": 2})
    with robot.sim(physics=True) as ctx:
        chat_loop(robot, mode="physics")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)


# -- Object spawning --------------------------------------------------------


def _spawn_manipulable_objects(
    robot: Geodude,
    objects: dict[str, int],
    fixture_types: set[str],
) -> None:
    """Scatter non-fixture objects on the worktop (simulates perception).

    Re-uses the same collision-free TSR placement as recycle.py.
    """
    import mujoco
    from asset_manager import AssetManager
    from prl_assets import OBJECTS_DIR
    from tsr.placement import TablePlacer

    specs = [(t, n) for t, n in objects.items() if t not in fixture_types]
    if not specs:
        return

    assets = AssetManager(str(OBJECTS_DIR))
    wt_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, "worktop")
    wt_size = robot.model.site_size[wt_id]
    worktop_pos = robot.data.site_xpos[wt_id].copy()

    placer = TablePlacer(wt_size[0] - 0.05, wt_size[1] - 0.05)
    table_surface = np.eye(4)
    table_surface[:3, 3] = worktop_pos

    for obj_type, count in specs:
        gp = assets.get(obj_type)["geometric_properties"]
        if gp["type"] == "cylinder":
            templates = placer.place_cylinder(gp["radius"], gp["height"])
        elif gp["type"] == "box":
            templates = placer.place_box(gp["size"][0], gp["size"][1], gp["size"][2])
        else:
            continue

        for _ in range(count):
            tsr = templates[0].instantiate(table_surface)
            pos = tsr.sample()[:3, 3]
            name = robot.env.registry.activate(obj_type, pos=list(pos))
            mujoco.mj_forward(robot.model, robot.data)

            body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id = robot.model.body_jntadr[body_id]
            qpos_adr = robot.model.jnt_qposadr[jnt_id]
            for _ in range(50):
                if not _has_object_collision(robot.model, robot.data, name):
                    break
                robot.data.qpos[qpos_adr : qpos_adr + 3] = tsr.sample()[:3, 3]
                mujoco.mj_forward(robot.model, robot.data)

    mujoco.mj_forward(robot.model, robot.data)


def _has_object_collision(model, data, body_name: str) -> bool:
    """Check if a body is in contact with any other object (not floor)."""
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if (b1 == body_id or b2 == body_id) and c.dist < 0:
            other = b2 if b1 == body_id else b1
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
            if other_name and other_name != "world":
                return True
    return False


# -- Scene presets -----------------------------------------------------------

SCENE_PRESETS = {
    "recycling": {
        "description": "Recycling demo (3 cans, 1 potted meat, 2 bins)",
        "objects": {"can": 3, "potted_meat_can": 1, "recycle_bin": 2},
        "fixtures": {
            "recycle_bin": [[0.85, -0.35, 0.01], [-0.85, -0.35, 0.01]],
        },
    },
}


def choose_scene() -> tuple[dict[str, int], dict[str, list[list[float]]]]:
    """Interactive scene selection. Returns (objects, fixtures)."""
    print("\nWhat scene would you like?\n")
    presets = list(SCENE_PRESETS.items())
    for i, (_, preset) in enumerate(presets, 1):
        print(f"  {i}. {preset['description']}")
    print(f"  {len(presets) + 1}. Custom\n")

    choice = input("> ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(presets):
            _, preset = presets[idx]
            return preset["objects"], preset["fixtures"]
    except ValueError:
        pass

    # Custom scene
    from prl_assets import OBJECTS_DIR

    available = sorted(
        d.name for d in OBJECTS_DIR.iterdir()
        if d.is_dir() and (d / "meta.yaml").exists()
    )
    print(f"\nAvailable objects: {', '.join(available)}")
    print('How many of each? (e.g. "4 cans, 2 recycle_bins")')
    spec = input("> ").strip()

    objects: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if len(tokens) == 2:
            count_str, obj_type = tokens
            # Strip trailing 's' for plurals
            if obj_type.endswith("s") and obj_type[:-1] in available:
                obj_type = obj_type[:-1]
            try:
                objects[obj_type] = int(count_str)
            except ValueError:
                print(f"  Skipping '{part}' — expected 'N type'")

    return objects, {}


# -- Tool definitions for LLM -----------------------------------------------


def _build_tools(robot: Geodude) -> list[dict]:
    """Build tool schemas for the LLM."""
    return [
        {
            "name": "pickup",
            "description": (
                "Pick up an object. Prefer omitting target — the planner automatically "
                "finds the nearest reachable object and the best arm. Only specify "
                "target when the user asks for a specific object by name or type. "
                "target can be a specific instance ('can_0'), a type ('can' matches "
                "any can), or omitted for any reachable object. "
                "arm can be 'left', 'right', or omitted to try both."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Object name or type. Omit to pick up nearest reachable object (preferred).",
                    },
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which arm to use. Omit to auto-select best arm (preferred).",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "place",
            "description": (
                "Place the held object at a destination (e.g. a bin). "
                "Auto-detects which arm is holding. Destination can be a specific "
                "instance ('recycle_bin_0'), a type ('recycle_bin' for any bin), "
                "or omitted for any available destination."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Destination name or type. Omit for any available destination.",
                    },
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which arm. Omit to auto-detect holding arm (preferred).",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "go_home",
            "description": "Return arms to ready position. Optionally specify which arm.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which arm, or omit for both",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_objects",
            "description": (
                "List all graspable objects in the scene with their positions. "
                "Returns object names and xyz positions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "holding",
            "description": "Check what the robot is holding. Returns (arm_side, object_name) or null.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "get_ft_wrench",
            "description": (
                "Read the 6-axis force/torque sensor at a wrist. "
                "Returns [fx, fy, fz, tx, ty, tz] in N and Nm. "
                "Only meaningful in physics mode."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Which arm's wrist sensor to read",
                    },
                },
                "required": ["arm"],
            },
        },
        {
            "name": "find_objects",
            "description": (
                "Find objects matching a type. E.g. find_objects('can') returns "
                "all cans. find_objects() returns all graspable objects."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Object type to search for (e.g. 'can'), or omit for all",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_robot_state",
            "description": (
                "Get detailed robot state: joint positions, EE pose, base height, "
                "gripper state, home status, and holding info for each arm."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "reset_scene",
            "description": (
                "Reset the scene: re-spawn all original objects at random positions "
                "on the worktop, release any grasped objects, and return arms to home. "
                "Use when the user wants to start over or put objects back."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    ]


def _execute_tool(robot: Geodude, name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "pickup":
        ok = robot.pickup(args.get("target"), arm=args.get("arm"))
        return f"{'Success' if ok else 'Failed'}: pickup({args})"

    elif name == "place":
        ok = robot.place(args.get("destination"), arm=args.get("arm"))
        return f"{'Success' if ok else 'Failed'}: place({args})"

    elif name == "go_home":
        ok = robot.go_home() if "arm" not in args else robot.go_home()
        # go_home doesn't take arm param at top level, use _ArmScope
        if "arm" in args:
            from geodude.primitives import go_home
            ok = go_home(robot, arm=args["arm"])
        return f"{'Success' if ok else 'Failed'}: go_home({args})"

    elif name == "get_objects":
        objects = robot.find_objects()
        result = {}
        for obj_name in objects:
            pose = robot.get_object_pose(obj_name)
            result[obj_name] = {
                "position": [round(float(x), 3) for x in pose[:3, 3]],
            }
        return json.dumps(result, indent=2)

    elif name == "holding":
        h = robot.holding()
        if h:
            return json.dumps({"arm": h[0], "object": h[1]})
        return "null — both arms are free"

    elif name == "get_ft_wrench":
        arm_obj = robot._resolve_arm(args["arm"])
        wrench = arm_obj.get_ft_wrench()
        return (
            f"[{', '.join(f'{v:.2f}' for v in wrench)}] "
            f"(force: {np.linalg.norm(wrench[:3]):.2f}N, "
            f"torque: {np.linalg.norm(wrench[3:]):.2f}Nm)"
        )

    elif name == "find_objects":
        objects = robot.find_objects(args.get("target"))
        return json.dumps(objects)

    elif name == "reset_scene":
        import mujoco

        # Hide all active objects
        for obj_name in list(robot.env.registry.active_objects()):
            robot.env.registry.hide(obj_name)
        # Release any grasped objects
        for obj in list(robot.grasp_manager.grasped.keys()):
            robot.grasp_manager.mark_released(obj)
        # Re-setup fixtures and spawn objects
        original_objects = args.get("_original_objects", {})
        original_fixtures = args.get("_original_fixtures", {})
        robot.setup_scene(fixtures=original_fixtures if original_fixtures else None)
        fixture_types = set(original_fixtures.keys()) if original_fixtures else set()
        _spawn_manipulable_objects(robot, original_objects, fixture_types)
        mujoco.mj_forward(robot.model, robot.data)
        n = len(robot.find_objects())
        return f"Success: scene reset with {n} objects"

    elif name == "get_robot_state":
        state = {}
        for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
            ee_pose = arm.get_ee_pose()
            current_q = [float(robot.data.qpos[i]) for i in arm.joint_qpos_indices]

            at_home = False
            if "ready" in robot.named_poses and side in robot.named_poses["ready"]:
                ready_q = np.array(robot.named_poses["ready"][side])
                at_home = bool(np.allclose(current_q, ready_q, atol=0.05))

            held = list(robot.grasp_manager.get_grasped_by(side))

            arm_state = {
                "ee_position": [round(float(x), 3) for x in ee_pose[:3, 3]],
                "joint_positions_rad": [round(float(q), 3) for q in current_q],
                "at_home": at_home,
                "holding": held[0] if held else None,
            }

            base = robot._get_base_for_arm(arm)
            if base is not None:
                arm_state["base_height_m"] = round(float(base.get_height()), 3)
                arm_state["base_range_m"] = [
                    round(float(base.height_range[0]), 3),
                    round(float(base.height_range[1]), 3),
                ]

            state[side] = arm_state

        return json.dumps(state, indent=2)

    else:
        return f"Unknown tool: {name}"


# -- Scene state for LLM context -------------------------------------------


def _scene_summary(robot: Geodude) -> str:
    """Build a scene state summary for the LLM system prompt."""
    lines = ["Current scene state:"]

    # Objects
    objects = robot.find_objects()
    if objects:
        lines.append(f"  Objects on table: {len(objects)}")
        for name in objects:
            pose = robot.get_object_pose(name)
            pos = pose[:3, 3]
            lines.append(f"    - {name} at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    else:
        lines.append("  No objects on table.")

    # Holding
    h = robot.holding()
    if h:
        lines.append(f"  Holding: {h[0]} arm has {h[1]}")
    else:
        lines.append("  Holding: nothing (both arms free)")

    # Arm state
    for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
        ee_pose = arm.get_ee_pose()
        pos = ee_pose[:3, 3]
        # Home status
        at_home = False
        if "ready" in robot.named_poses and side in robot.named_poses["ready"]:
            ready_q = np.array(robot.named_poses["ready"][side])
            current_q = np.array([robot.data.qpos[i] for i in arm.joint_qpos_indices])
            at_home = np.allclose(current_q, ready_q, atol=0.05)
        # Holding
        held = list(robot.grasp_manager.get_grasped_by(side))
        holding_str = f", holding {held[0]}" if held else ""
        # Base height
        base = robot._get_base_for_arm(arm)
        base_str = f", base at {base.get_height():.2f}m" if base else ""
        status = "at home" if at_home else "not at home"
        lines.append(
            f"  {side.capitalize()} arm: EE=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] "
            f"({status}{holding_str}{base_str})"
        )

    return "\n".join(lines)


# -- Main chat loop ---------------------------------------------------------

SYSTEM_PROMPT = """\
You are the control interface for Geodude, a bimanual robot.

## Robot description
- Two UR5e 6-DOF arms ("left" and "right") with Robotiq 2F-140 parallel-jaw grippers
- Each arm is mounted on a Vention linear actuator base that moves vertically (0.0–0.5m)
- The left arm's workspace is the left side of the table (negative x), the right arm covers the right side (positive x). There is some overlap in the center.
- Objects near x=0 may be reachable by either arm
- The "home" or "ready" position has both arms raised with grippers above the worktop
- After a successful pickup, the base lifts 15cm to clear clutter

## Grippers
- Each gripper is a parallel-jaw gripper that can grasp objects by closing on them
- After placing an object, it is removed from the scene (simulates dropping into a bin or container)
- To bring objects back, use reset_scene — this re-spawns all original objects at new random positions
- The gripper is either open or closed — there is no partial close

## Objects and destinations
- Objects are graspable things on the worktop (cans, potted meat cans, etc.)
- Destinations are stationary fixtures like recycle bins — they are NOT graspable
- Object names follow the pattern: type_N (e.g. "can_0", "can_1", "potted_meat_can_0", "recycle_bin_0")
- To refer to any object of a type, use just the type name (e.g. "can" matches any can)

## Sensors
- Each arm has a 6-axis force/torque sensor at the wrist
- F/T readings are only meaningful in physics mode — in kinematic mode they are near-zero noise
- Use F/T to detect contact, estimate object weight, or monitor grip

## Tool usage principles
- Let the planner do its job. When the user says "clear the table" or "pick up a can", call pickup() with NO target and NO arm — the planner finds the nearest reachable object and the best arm automatically. Only specify target when the user names a specific object (e.g. "pick up can_0" or "pick up the potted meat").
- Same for place(): omit destination and arm unless the user specifies one.
- For repetitive tasks ("clear the table"), call pickup() then place() in a loop until pickup fails (nothing left).

## User environment
The user is in an IPython console with `robot`, `ctx`, and `np` available. When the user \
asks how to do something in Python, give them the actual Python API — not tool names. \
ONLY recommend methods that actually exist on the robot object. The complete list:

- `robot.find_objects()` — list all graspable objects (returns list of names)
- `robot.find_objects("can")` — find objects matching a type
- `robot.holding()` — what is the robot holding? Returns (side, name) or None
- `robot.pickup()` — pick up nearest reachable object
- `robot.pickup("can")` — pick up any can
- `robot.pickup("can_0")` — pick up specific object
- `robot.place("recycle_bin")` — place held object in any bin
- `robot.go_home()` — return arms to ready position
- `robot.left_arm.get_ft_wrench()` — read left wrist F/T sensor
- `robot.right_arm.get_ft_wrench()` — read right wrist F/T sensor
- `robot.get_object_pose("can_0")` — get 4x4 pose matrix of an object
- `robot.left_arm.get_ee_pose()` — get left end-effector 4x4 pose

Lower-level APIs (available but not commonly needed):

- `robot.env.registry.hide("can_0")` — remove an object from the scene (disable physics + rendering)
- `robot.env.registry.activate("can", pos=[x, y, z])` — spawn/show an object at a position
- `robot.env.registry.active_objects()` — list all currently active object names
- `robot.env.registry.is_active("can_0")` — check if an object is in the scene
- `robot.grasp_manager.get_grasped_by("left")` — set of objects grasped by an arm
- `robot.grasp_manager.mark_released("can_0")` — release a grasped object
- `robot.left_arm.get_joint_positions()` — current joint angles (6,)
- `robot.left_arm.get_joint_limits()` — (lower, upper) arrays
- `robot.left_base.get_height()` — current base height in meters
- `robot.left_base.set_height(0.3)` — teleport base to height (no physics)
- `robot.data` — raw MuJoCo data (qpos, qvel, sensordata, etc.)
- `robot.model` — raw MuJoCo model

NEVER recommend methods that don't exist. If unsure whether a method exists, say so. \
Tool names (pickup, get_objects, get_ft_wrench, etc.) are for YOUR internal use only — \
never tell the user to call tool names directly.

## Rules
- Use tools to act. Don't describe what you would do — do it.
- After each action, briefly report the result (1 sentence).
- For spatial queries ("closest", "nearest"), compute Euclidean distances from arm EE positions to object positions using the coordinates in the scene state.
- If a request is ambiguous, ask for clarification.
- If an action fails, explain what happened and suggest alternatives (e.g. try the other arm).
- Never make up information. If you don't know something, say so or use a tool to find out.
- Keep responses concise — no filler, no restating the question.

{scene_state}
"""


class ChatSession:
    """Manages LLM conversation state for robot control.

    Holds the message history, tool definitions, and scene config.
    Call send() for each user message — usable from any REPL.
    """

    def __init__(
        self,
        robot: Geodude,
        *,
        mode: str = "kinematic",
        model_name: str = "claude-sonnet-4-20250514",
        original_objects: dict[str, int] | None = None,
        original_fixtures: dict[str, list[list[float]]] | None = None,
    ):
        import anthropic

        self.robot = robot
        self.mode = mode
        self.model_name = model_name
        self.original_objects = original_objects or {}
        self.original_fixtures = original_fixtures or {}
        self.client = anthropic.Anthropic()
        self.tools = _build_tools(robot)
        self.messages: list[dict] = []

    def send(self, user_input: str) -> str:
        """Send a message to the LLM and execute any tool calls.

        Returns the final text response.
        """
        self.messages.append({"role": "user", "content": user_input})
        response_text = ""

        try:
            response_text = self._run_conversation()
        except Exception as e:
            # On error, remove the failed user message and reset cleanly
            logger.warning("Chat error: %s. Resetting conversation.", e)
            self.messages.clear()
            response_text = f"Error: {e}. Conversation reset."

        return response_text

    def _run_conversation(self) -> str:
        """Inner conversation loop. Raises on API errors."""
        response_text = ""

        while True:
            system = SYSTEM_PROMPT.format(
                scene_state=_scene_summary(self.robot)
            )

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                system=system,
                tools=self.tools,
                messages=self.messages,
            )

            assistant_content = response.content
            self.messages.append({"role": "assistant", "content": assistant_content})

            tool_calls = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_calls:
                for block in assistant_content:
                    if hasattr(block, "text"):
                        response_text += block.text
                break

            # Execute tools — always produce a result for each call
            tool_results = []
            for tc in tool_calls:
                print(f"  \u2192 {tc.name}({json.dumps(tc.input)})")
                try:
                    args = dict(tc.input)
                    if tc.name == "reset_scene":
                        args["_original_objects"] = self.original_objects
                        args["_original_fixtures"] = self.original_fixtures
                    result = _execute_tool(self.robot, tc.name, args)
                except Exception as e:
                    result = f"Error executing {tc.name}: {e}"
                status = "\u2713" if "Success" in result or "Error" not in result else "\u2717"
                print(f"  {status} {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            # Append updated scene state to last tool result
            tool_results[-1]["content"] += (
                f"\n\nUpdated scene:\n{_scene_summary(self.robot)}"
            )

            self.messages.append({"role": "user", "content": tool_results})

        return response_text
