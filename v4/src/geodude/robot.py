"""Geodude's robot API: everyday calls over ssrobot and ssgrasping, with Geodude's defaults
(personalrobotics/geodude#209).

``Geodude`` binds to an open ``RobotContext`` and wires, once, what a program would
otherwise repeat: each arm's planners and the names its package declares, the object
registration, the driving of each skill on the context's clock, and Geodude's recovery
(``recycling.recover``). It owns policy: which arm, which object, which bin. Planning,
execution, observation and the skills stay ssrobot's and ssgrasping's, and every
``recycling`` function stays available beneath it.

    with RobotContext(package.description, recycling.runtime(package, scene),
                      object_poses=recycling.OBJECT_POSES) as ctx:
        robot = Geodude(ctx, package)
        if robot.pickup("can"):
            robot.place()  # into the bin on the holding arm's side
        robot.go_home()

Abort (``request_abort``) cancels the active skill (``SkillExecution.cancel``) or stops the
active command, and leaves what the robot holds as it is: the result reports it, and
``holding`` still does. Nothing starts again until ``clear_abort``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ssgrasping import (
    Candidate,
    Candidates,
    GraspSpec,
    SkillExecution,
    SkillResult,
    Stage,
    drop,
    obstructing,
    run_together,
)
from ssrobot import (
    Execution,
    ExecutionState,
    GripperCommand,
    GripperTermination,
    ObservationRequest,
    Planner,
    Pose,
    RobotContext,
    RobotPackage,
    ValidationError,
)

from geodude import outcomes, recycling

SIDES = tuple(recycling.BIN_OF)
"""Geodude's arms, by side."""
HOME = "ready"
"""The package's home configuration."""
AT_GOAL_RAD = 1e-3
"""How near its goal, in every joint, an arm already is when ``go_to`` leaves it there."""
ABORTED = "aborted by request"


@dataclass(frozen=True)
class TaskResult:
    """What a call did. True when it succeeded.

    ``held`` is what the deciding stage believed of the object (ssgrasping's conservative
    ``Stage.held``: None when it could not tell), ``attached`` whether the scene declares
    it in the hand now. ``results`` are the skills' own results, ``recovery`` what
    Geodude's recovery did after a failure."""

    succeeded: bool
    detail: str
    object: str | None = None
    arm: str | None = None
    held: bool | None = None
    attached: bool = False
    aborted: bool = False
    results: tuple[SkillResult, ...] = ()
    recovery: outcomes.Recovery | None = None

    def __bool__(self) -> bool:
        return self.succeeded


@dataclass
class _Held:
    """An object an arm picked: how it was grasped, and where from, for its recovery."""

    object: str
    spec: GraspSpec
    origin: tuple[float, float]


class Geodude:
    """Geodude in an open ``RobotContext`` (see the module). ``tick`` is called after each
    step of the clock (a viewer refresh); ``read`` registers each object's pose, as
    ``recycling.register`` does. Construction registers the objects on the table, builds
    each arm's planners, and builds both planning scenes."""

    def __init__(
        self,
        ctx: RobotContext,
        package: RobotPackage,
        *,
        tick: Callable[[], None] = lambda: None,
        read: recycling.Read = recycling.pose_stub,
    ) -> None:
        self._ctx = ctx
        self._package = package
        self._tick = tick
        self._read = read
        self._abort = threading.Event()
        self._skill: SkillExecution | None = None
        self._held: dict[str, _Held] = {}
        self.left = GeodudeArm(self, "left")
        self.right = GeodudeArm(self, "right")
        recycling.register(ctx, recycling.on_table(ctx), read)
        recycling.warm_planning(ctx, ctx.snapshot(), [a.arm for a in self.arms])

    @property
    def ctx(self) -> RobotContext:
        return self._ctx

    @property
    def arms(self) -> tuple[GeodudeArm, GeodudeArm]:
        return self.right, self.left

    def arm(self, side: str) -> GeodudeArm:
        if side not in SIDES:
            raise ValueError(f"no arm {side!r}; one of {SIDES}")
        return self.right if side == "right" else self.left

    # Abort.

    def request_abort(self) -> None:
        """Cancel the active skill or command at the next step, and refuse every call until
        ``clear_abort``. Safe from another thread."""
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    # Lookups.

    def find_objects(self, target: str | None = None) -> list[str]:
        """The objects on the table, neither held nor removed as landed, that ``target``
        names: an instance (``can_0``), a type (``can``), or anything (None)."""
        return [
            name
            for name in recycling.on_table(self._ctx)
            if target is None or target in (name, recycling.kind_of(name))
        ]

    def holding(self) -> tuple[str, str] | None:
        """``(side, object)`` for an object the scene declares in a hand, the right first."""
        for a in self.arms:
            if (obj := a.holding()) is not None:
                return a.side, obj
        return None

    # Tasks.

    def pickup(self, target: str | None = None, *, arm: str | None = None) -> TaskResult:
        """Pick up one object that ``target`` names (``find_objects``) with a free hand:
        ``arm``'s, or either, the arm on the object's side preferred. Every such object
        and arm is raced at once (ssgrasping's ``Candidates``): the first that plans by
        priority, then cost, is grasped and lifted. A failure with the object still in
        the hand is recovered (``recycling.recover``)."""
        if self.aborted:
            return TaskResult(False, ABORTED, aborted=True)
        allowed = self.arms if arm is None else (self.arm(arm),)
        free = [a for a in allowed if a.holding() is None]
        if not free:
            return TaskResult(False, "no free hand", arm=arm)
        recycling.register(self._ctx, recycling.on_table(self._ctx), self._read)
        found = self.find_objects(target)
        refused = {o: why for o in found if (why := recycling.ungraspable(self._ctx, o))}
        objects = [o for o in found if o not in refused]
        if not objects:
            why = "".join(f"; {o}: {w}" for o, w in refused.items())
            return TaskResult(False, f"nothing graspable matches {target!r}{why}")
        candidates = []
        for obj in objects:
            preferred = recycling.arm_for(recycling.registered_pose(self._ctx, obj).position)
            for a in free:
                planner, checker = a.planners
                candidates.append(
                    Candidate(
                        label=f"{a.side} {obj}",
                        grasp=recycling.grasp_spec(self._ctx, obj, a.arm),
                        planner=planner,
                        checker=checker,
                        priority=recycling.PREFERRED if a.side == preferred.side else 0,
                    )
                )
        race = Candidates(self._ctx, candidates)
        run_together(self._ctx, [race], max_ticks=recycling.MAX_TICKS, tick=self._stepped)
        selection = race.selection
        if selection is None:
            return TaskResult(False, f"no grasp of {', '.join(objects)} planned")
        side, obj = selection.candidate.label.split(" ")
        chosen = self.arm(side)
        origin = recycling.registered_pose(self._ctx, obj).position
        picked = self._run(self._ctx, selection.grasp(self._ctx), self._stepped)
        spec = selection.candidate.grasp
        if picked.succeeded:
            self._held[side] = _Held(obj, spec, (float(origin[0]), float(origin[1])))
        return self._finish(chosen, obj, picked, spec)

    def place(
        self,
        destination: str | None = None,
        *,
        arm: str | None = None,
        at: tuple[float, float] | None = None,
    ) -> TaskResult:
        """Place what ``arm``'s hand holds (``holding``'s, by default): into the bin on its
        side (None, ``recycle_bin`` or that bin's name; each arm reaches only its own), or
        upright on the worktop (``worktop``) at ``at``, where it was picked from by
        default. A failure with the object still in the hand is recovered."""
        if self.aborted:
            return TaskResult(False, ABORTED, aborted=True)
        if arm is None:
            held = self.holding()
            if held is None:
                return TaskResult(False, "nothing held")
            arm = held[0]
        chosen = self.arm(arm)
        obj = chosen.holding()
        if obj is None:
            return TaskResult(False, "nothing held", arm=arm)
        bin_name, _ = chosen.arm.bin
        if destination not in (None, "recycle_bin", bin_name, "worktop"):
            raise ValueError(
                f"no destination {destination!r} for the {arm} arm; "
                f"one of None, 'recycle_bin', {bin_name!r}, 'worktop'"
            )
        record = self._held.get(arm)
        if record is None or record.object != obj:
            pose = recycling.registered_pose(self._ctx, obj).position
            record = _Held(
                obj, recycling.grasp_spec(self._ctx, obj, chosen.arm), (pose[0], pose[1])
            )
        planner, checker = chosen.planners
        if destination == "worktop":
            placed = recycling.place_on_worktop(
                self._ctx,
                self._package,
                obj,
                at or record.origin,
                arm=chosen.arm,
                planners=chosen.planners,
                tick=self._stepped,
                read=self._read,
                run=self._run,
            )
            return self._finish(chosen, obj, placed, record.spec)
        spec = recycling.drop_spec(obj, chosen.arm)
        if spec is None:
            return TaskResult(
                False, f"undroppable: {recycling.undroppable(obj)}", obj, arm, True, True
            )
        dropping = drop(
            self._ctx,
            planner,
            spec,
            check_planner=checker,
            released=recycling.assume_landed(self._ctx, obj),
        )
        dropped = self._run(self._ctx, dropping, self._stepped)
        return self._finish(chosen, obj, dropped, record.spec, origin=record.origin)

    def go_to(self, name: str, *, arm: str | None = None) -> TaskResult:
        """Move ``arm``, or each arm in turn, the right first, to its joints' positions in
        the package's named configuration ``name``. The lifts stay where they are, and
        what a hand holds is carried."""
        if self.aborted:
            return TaskResult(False, ABORTED, aborted=True)
        configuration = self._ctx.description.configuration(name)
        joints = self._ctx.description.group(configuration.group).joints
        target = dict(zip(joints, configuration.positions, strict=True))
        details = []
        for a in self.arms if arm is None else (self.arm(arm),):
            reached = a.move_to(tuple(target[j] for j in a.joints))
            details.append(f"{a.side}: {reached}")
            if reached != ExecutionState.SUCCEEDED.value:
                return TaskResult(False, "; ".join(details), arm=a.side, aborted=self.aborted)
        return TaskResult(True, "; ".join(details), arm=arm)

    def go_home(self, *, arm: str | None = None) -> TaskResult:
        """``go_to`` the package's home (``HOME``)."""
        return self.go_to(HOME, arm=arm)

    # Driving.

    def _stepped(self) -> None:
        """After each step: the caller's tick, then, if abort was requested, cancel the
        active skill, or stop whatever else is executing."""
        self._tick()
        if not self.aborted:
            return
        if self._skill is not None:
            self._skill.cancel(ABORTED)
        else:
            self._ctx.stop()

    def _run(
        self, ctx: RobotContext, execution: SkillExecution, tick: Callable[[], None]
    ) -> SkillResult:
        """``recycling.drive``, with ``execution`` the skill an abort cancels."""
        self._skill = execution
        try:
            if self.aborted:
                execution.cancel(ABORTED)
            return recycling.drive(ctx, execution, tick)
        finally:
            self._skill = None

    def _wait(self, execution: Execution) -> str:
        """Step until ``execution`` ends; its end state."""
        if self.aborted:
            self._ctx.cancel(execution)
        return recycling.step_until(self._ctx, execution, self._stepped).state.value

    def _finish(
        self,
        chosen: GeodudeArm,
        obj: str,
        result: SkillResult,
        spec: GraspSpec,
        *,
        origin: tuple[float, float] | None = None,
    ) -> TaskResult:
        """The result of a skill on ``obj``, recovered first when it failed with ``obj``
        still in the hand and was not aborted. ``origin``, for a drop: where to put the
        object back if no release planned (``recycling.recover``)."""
        failed: Stage | None = recycling.failed_stage(result)
        decided = failed or result.stage
        recovery = None
        aborted = self.aborted
        if failed is not None and not aborted and chosen.holding() == obj:
            planner, checker = chosen.planners
            recovery = recycling.recover(
                self._ctx,
                obj,
                chosen.arm,
                failed,
                spec,
                planner,
                self._stepped,
                self._read,
                package=self._package,
                checker=checker,
                origin=origin,
            )
        attached = chosen.holding() == obj
        if not attached:
            self._held.pop(chosen.side, None)
        if failed is None:
            outcome = "succeeded"
        else:
            outcome = "failed" if failed.reason is None else failed.reason.value
        detail = f"{result.skill}.{decided.name}: {outcome}"
        if recovery is not None:
            detail += f"; {recovery.detail}"
        return TaskResult(
            succeeded=failed is None,
            detail=detail,
            object=obj,
            arm=chosen.side,
            held=decided.held,
            attached=attached,
            aborted=aborted,
            results=(result,),
            recovery=recovery,
        )


@dataclass
class GeodudeArm:
    """One of Geodude's arms, its components named once (``recycling.Arm``), checked
    against the package's description, and its planners built once."""

    robot: Geodude
    side: str
    arm: recycling.Arm = field(init=False)
    planners: tuple[Planner, Planner] = field(init=False, repr=False)
    _tare: tuple[float, ...] = field(init=False, repr=False, default=(0.0,) * 6)

    def __post_init__(self) -> None:
        self.arm = recycling.Arm(self.side)
        description = self.robot.ctx.description
        description.manipulator(self.arm.manipulator)  # each fails if the package lacks it
        description.end_effector(self.arm.hand)
        description.gripper(self.arm.gripper)
        description.channel(self.arm.opening)
        description.channel(self.arm.wrench)
        self.planners = recycling.planners(self.robot._package, self.arm)

    @property
    def joints(self) -> tuple[str, ...]:
        return tuple(self.robot.ctx.description.group(self.arm.manipulator).joints)

    # State.

    def joint_positions(self) -> tuple[float, ...]:
        """The arm's joint positions now."""
        snapshot = self.robot.ctx.snapshot()
        return tuple(snapshot.position(j) for j in self.joints)

    def end_effector_pose(self) -> Pose:
        """The hand's frame, by forward kinematics at the arm's joint positions now."""
        ctx = self.robot.ctx
        frame = ctx.description.end_effector(self.arm.hand).frame
        snapshot = ctx.snapshot()
        q = tuple(snapshot.position(j) for j in self.joints)
        return ctx.planning_scene(snapshot, self.arm.manipulator).forward_kinematics(q, frame)

    def wrench(self) -> tuple[float, ...]:
        """The wrist's force and torque now, less the reading at the last ``tare``."""
        raw = self._wrench()
        return tuple(r - t for r, t in zip(raw, self._tare, strict=True))

    def tare(self) -> None:
        """Take the wrist's reading now as its zero."""
        self._tare = self._wrench()

    def _wrench(self) -> tuple[float, ...]:
        observed = self.robot.ctx.observe(ObservationRequest(channels=(self.arm.wrench,)))
        value = observed.reading(self.arm.wrench).value
        assert isinstance(value, tuple)
        return tuple(float(v) for v in value)

    def holding(self) -> str | None:
        """The object the scene declares in this hand."""
        for attachment in self.robot.ctx.scene.attachments:
            if attachment.end_effector == self.arm.hand:
                return attachment.object
        return None

    # Gripper.

    def open(self) -> TaskResult:
        """Open the fingers, unless the hand holds an object (``place`` it), or the fingers,
        opened, would touch something (ssgrasping's ``obstructing``)."""
        if (obj := self.holding()) is not None:
            return TaskResult(False, f"holding {obj}: place it", obj, self.side, True, True)
        ctx = self.robot.ctx
        try:
            touched = obstructing(
                ctx, ctx.snapshot(), self.arm.manipulator, self.arm.gripper, "open", ()
            )
        except ValidationError as e:
            return TaskResult(
                False, f"the fingers' end could not be checked: {e.code}", arm=self.side
            )
        if touched:
            return TaskResult(
                False, f"opened, the fingers would touch {', '.join(touched)}", arm=self.side
            )
        return self._grip(1.0)

    def close(self) -> TaskResult:
        """Close the fingers. Nothing is attached: ``pickup`` grasps."""
        return self._grip(0.0)

    def _grip(self, opening: float) -> TaskResult:
        if self.robot.aborted:
            return TaskResult(False, ABORTED, arm=self.side, aborted=True)
        ctx = self.robot.ctx
        execution = ctx.submit(GripperCommand(gripper=self.arm.gripper, opening=opening))
        state = self.robot._wait(execution)
        status = execution.status.gripper
        ended = (
            "" if status is None else f", {status.termination.value} at {status.final_opening:.3f}"
        )
        # Opening must reach open; closing may stop short, on whatever is between the fingers.
        succeeded = (
            state == ExecutionState.SUCCEEDED.value
            and status is not None
            and (opening == 0.0 or status.termination is GripperTermination.REACHED_TARGET)
        )
        return TaskResult(succeeded, f"{state}{ended}", arm=self.side, aborted=self.robot.aborted)

    # Motion.

    def move_to(self, q: Sequence[float]) -> str:
        """Plan to the joint positions ``q`` and execute: the end state, or why it did not
        plan. An arm already within ``AT_GOAL_RAD`` of ``q`` in every joint is not moved
        (ssrobot#253: planning to where it is raises)."""
        ctx = self.robot.ctx
        if all(abs(a - b) <= AT_GOAL_RAD for a, b in zip(self.joint_positions(), q, strict=True)):
            return ExecutionState.SUCCEEDED.value
        planner, _ = self.planners
        plan = ctx.manipulator(self.arm.manipulator).plan_to_configuration(q, planner=planner)
        if not plan.succeeded:
            return f"not planned: {plan.status.value}"
        return self.robot._wait(ctx.execute(plan, planner=planner))

    # The robot's tasks, with this arm.

    def pickup(self, target: str | None = None) -> TaskResult:
        return self.robot.pickup(target, arm=self.side)

    def place(
        self, destination: str | None = None, *, at: tuple[float, float] | None = None
    ) -> TaskResult:
        return self.robot.place(destination, arm=self.side, at=at)

    def go_home(self) -> TaskResult:
        return self.robot.go_home(arm=self.side)
