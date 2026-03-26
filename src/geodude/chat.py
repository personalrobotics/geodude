"""LLM chat integration for Geodude robot control.

Provides ChatSession for natural language → tool-use robot control.
Scene setup and demo infrastructure live in demo_loader.py.

Example::

    from geodude.demo_loader import setup_robot
    robot = setup_robot({"can": 4, "recycle_bin": 2}, {})
    with robot.sim(physics=True) as ctx:
        session = ChatSession(robot, mode="physics")
        print(session.send("pick up a can"))
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geodude.robot import Geodude

logger = logging.getLogger(__name__)


# -- Tool definitions for LLM -----------------------------------------------


def _build_tools() -> list[dict]:
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


def _execute_tool(
    robot: Geodude,
    name: str,
    args: dict,
    *,
    original_objects: dict[str, int] | None = None,
    original_fixtures: dict[str, list[list[float]]] | None = None,
) -> str:
    """Execute a tool call and return the result as a string."""
    if name == "pickup":
        ok = robot.pickup(args.get("target"), arm=args.get("arm"))
        return f"{'Success' if ok else 'Failed'}: pickup({args})"

    elif name == "place":
        ok = robot.place(args.get("destination"), arm=args.get("arm"))
        return f"{'Success' if ok else 'Failed'}: place({args})"

    elif name == "go_home":
        from geodude.primitives import go_home
        ok = go_home(robot, arm=args.get("arm"))
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
        original_objects = original_objects or {}
        original_fixtures = original_fixtures or {}
        robot.reset()
        fixture_types = set(original_fixtures.keys()) if original_fixtures else set()
        from geodude.demo_loader import _spawn_manipulable_objects
        _spawn_manipulable_objects(robot, original_objects, fixture_types)
        n = len(robot.find_objects())
        return f"Success: scene reset with {n} objects"

    elif name == "get_robot_state":
        state = {}
        for side, arm in [("left", robot.left_arm), ("right", robot.right_arm)]:
            ee_pose = arm.get_ee_pose()
            current_q = [float(robot.data.qpos[i]) for i in arm.joint_qpos_indices]

            at_home = _is_arm_at_home(robot, side, arm)

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


def _is_arm_at_home(robot: Geodude, side: str, arm) -> bool:
    """Check if an arm is at its home (ready) configuration."""
    if "ready" not in robot.named_poses or side not in robot.named_poses["ready"]:
        return False
    ready_q = np.array(robot.named_poses["ready"][side])
    current_q = np.array([robot.data.qpos[i] for i in arm.joint_qpos_indices])
    return bool(np.allclose(current_q, ready_q, atol=0.05))


def _api_reference(robot: Geodude) -> str:
    """Generate API reference from live docstrings."""
    import inspect

    from mj_environment import ObjectRegistry
    from mj_manipulator import Arm

    from geodude.vention_base import VentionBase

    lines = []

    def _add_section(title: str, cls, prefix: str):
        lines.append(f"\n{title}:")
        for name, method in sorted(inspect.getmembers(cls, predicate=inspect.isfunction)):
            if name.startswith("_"):
                continue
            try:
                sig = inspect.signature(method)
            except (ValueError, TypeError):
                continue
            doc = (inspect.getdoc(method) or "").split("\n")[0]
            params = [p for p in sig.parameters if p != "self"]
            param_str = ", ".join(params)
            lines.append(f"  {prefix}.{name}({param_str}) — {doc}")

    _add_section("robot (Geodude)", type(robot), "robot")
    _add_section("robot.left_arm / robot.right_arm (Arm)", Arm, "arm")
    _add_section("robot.env.registry (ObjectRegistry)", ObjectRegistry, "robot.env.registry")
    if robot.left_arm.gripper is not None:
        _add_section(
            "robot.left_arm.gripper / robot.right_arm.gripper (RobotiqGripper)",
            type(robot.left_arm.gripper), "gripper",
        )
    if robot.left_base is not None:
        _add_section("robot.left_base / robot.right_base (VentionBase)", VentionBase, "base")

    # Add key properties
    lines.append("\nKey properties:")
    lines.append("  robot.left_arm, robot.right_arm — Arm instances")
    lines.append("  robot.left_base, robot.right_base — VentionBase instances (or None)")
    lines.append("  robot.left, robot.right — scoped shortcuts (robot.left.pickup('can'))")
    lines.append("  robot.env — MuJoCo Environment")
    lines.append("  robot.env.registry — ObjectRegistry for hide/show/activate")
    lines.append("  robot.model — raw MuJoCo mjModel")
    lines.append("  robot.data — raw MuJoCo mjData (qpos, qvel, sensordata, etc.)")
    lines.append("  robot.grasp_manager — GraspManager for grasp state queries")

    return "\n".join(lines)


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
        at_home = _is_arm_at_home(robot, side, arm)
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

## Console and viewer
- The user is already inside a simulation context (`ctx` is available). Do NOT tell them to create a new `robot.sim()` context.
- The MuJoCo viewer is a launch-time option: the user must restart with `--viewer` flag (e.g. `uv run mjpython -m geodude.cli --demo recycling --viewer`). It cannot be opened mid-session.
- After modifying the scene (hide, activate, set_height, etc.), always call `robot.forward()` then `ctx.sync()` to update the viewer
- `robot.forward()` runs MuJoCo forward kinematics to update internal state
- `ctx.sync()` pushes the updated state to the viewer for rendering
- Example: `robot.env.registry.hide('can_0'); robot.forward(); ctx.sync()`

## Demos
- Demos are Python files in `src/geodude/demos/` with a `scene` dict and optional functions.
- `demos()` lists available demos. `save_demo('name')` saves the current scene + user-defined functions as a new demo.
- Demo functions (like `sort_all()`) are loaded into the console namespace automatically.
- When the user asks you to write a function, give them clean Python code they can paste into the console, test, and then save with `save_demo()`.
- `robot` is available as a global in demo files — no need to pass it as a parameter.

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
asks how to do something in Python, give them the actual Python API — not tool names.

NEVER recommend methods that don't exist. If unsure whether a method exists, say so. \
Tool names (pickup, get_objects, get_ft_wrench, etc.) are for YOUR internal use only — \
never tell the user to call tool names directly.

The complete API reference (auto-generated from docstrings):
{api_reference}

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
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: uv sync --extra chat"
            ) from None

        self.robot = robot
        self.mode = mode
        self.model_name = model_name
        self.original_objects = original_objects or {}
        self.original_fixtures = original_fixtures or {}
        self.client = anthropic.Anthropic()
        self.tools = _build_tools()
        self.api_reference = _api_reference(robot)
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

        scene_state = _scene_summary(self.robot)

        while True:
            system = SYSTEM_PROMPT.format(
                scene_state=scene_state,
                api_reference=self.api_reference,
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
                    result = _execute_tool(
                        self.robot, tc.name, dict(tc.input),
                        original_objects=self.original_objects,
                        original_fixtures=self.original_fixtures,
                    )
                except Exception as e:
                    result = f"Error executing {tc.name}: {e}"
                status = "\u2713" if "Success" in result or "Error" not in result else "\u2717"
                print(f"  {status} {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

            # Refresh scene state after tool execution
            scene_state = _scene_summary(self.robot)
            tool_results[-1]["content"] += f"\n\nUpdated scene:\n{scene_state}"

            self.messages.append({"role": "user", "content": tool_results})

        return response_text
