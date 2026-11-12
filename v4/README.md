# Geodude 4.0

Geodude is a bimanual robot: two UR5e arms, each on a Vention linear rail and fitted
with a Robotiq 2F-140 gripper, working over a table. Geodude 4.0 rebuilds its
application on [ssrobot](https://github.com/personalrobotics/ssrobot) and
[ssgrasping](https://github.com/personalrobotics/ssgrasping). It is developed here,
alongside the 3.x application at the repository root, and will replace it once it
covers the same tasks.

## What the application owns

A robot application is where the decisions specific to one robot and one task belong.
Geodude 4.0 is deliberately small because most of the work happens in shared layers,
each of which owns one kind of decision:

| Layer | Owns |
| --- | --- |
| [geodude_assets](https://github.com/personalrobotics/geodude_assets) | Geodude's robot package: its MuJoCo model, groups, end effectors, grippers, channels, and limits |
| ssrobot | The session, snapshots, planning contracts, the checked execution boundary, and the run's evidence |
| ssgrasping | Grasp, place, and drop skills written against robot-neutral contracts, candidate races, and planning ahead |
| [prl_assets](https://github.com/personalrobotics/prl_assets) and [sstsr](https://github.com/personalrobotics/sstsr) | Object geometry and models, and the gripper and placement templates built from them |
| Geodude 4.0 | Which arm takes which object, where the bins stand, the regions offered to each skill, the speeds, forces, and tolerances, how a scene is staged, how the run is driven, and what counts as success |

Geodude 4.0 owns the application's orchestration. It chooses and constructs its
planners, builds the specs it passes to each skill, decides which skills run in which
order, and advances the `RobotContext`, stepping its clock while skills and races
proceed. It does not contain a planner implementation, a low-level controller, a
simulator integration, or a grasping skill. ssrobot's runtime integrates MuJoCo,
sscbirrt plans, and ssgrasping implements the skills. The same separation is what allows
ssgrasping's skills to be written against robot-neutral contracts.

## The recycling demo

The demo stages cans upright on the worktop with two recycling bins beside the table,
one on each arm's side, then picks each can and drops it into a bin. With one can, the
arm on the can's side of the worktop picks it and drops it into its own bin.

With several cans, both arms race for each one. Each arm's whole feasibility prefix,
from the pregrasp through the transfer to its own bin's drop region, is planned before
anything moves. Closing the fingers, releasing the can, and where it comes to rest are
observed when they happen, not planned.
The arm on the can's side is preferred, and the other arm takes the can when the
preferred arm's prefix does not plan. The race for the next can is planned while the
current can is carried, on the prediction of that transfer. The prediction expects the
can to be released over the bin and then removed from the scene as landed, and the
selected pregrasp continues once the release has happened. The demo can also place a single can
upright at a target on the worktop under a force-guarded descent, instead of dropping
it.

The legacy demo's round stages five objects drawn, by the seed, from its nine types:
`can`, `spam_can`, `cracker_box`, `gelatin_box`, `sugar_box`, `pocky_box`,
`pop_tarts_case`, `fuze_bottle` and `plastic_glass` (`recycling.MIX_KINDS`).
`recycling.sort_all` sorts them all into the bins, racing both arms as above. An attempt
that fails is recovered only where releasing the object is safe: a lift that did not
plan releases it where it was grasped, and an object whose gripper opening showed it
leave the hand during a carry, which then stopped and reported it not held, is
detached: nothing is left in the hand. A held object whose drop's transfer does not plan
is put back on the worktop where it was picked from. An object released back onto the
worktop upright is tried once more. Anything else stops the round, the object still
declared held: the fingers open only onto the object's known support or an empty hand.
That includes a transfer whose command fails mid-carry and a release that is not
confirmed (#210). The table is clear once every object rests in a bin.

## The robot API

`geodude.robot.Geodude` wraps the same pieces for a program of its own (#209). It binds to
an open `RobotContext`, and wires once what each call would otherwise repeat: each arm's
planners and component names, the object registration, the driving of each skill on the
context's clock, and the recovery above.

```python
from ssrobot import RobotContext, load_installed_package
from geodude import recycling
from geodude.robot import Geodude
from geodude.scene import stage_scene

package = load_installed_package("geodude_assets")
scene = stage_scene("scene", package, objects={"can": 2}, fixtures=recycling.BINS, seed=0)
with RobotContext(
    package.description, recycling.runtime(package, scene), object_poses=recycling.OBJECT_POSES
) as ctx:
    robot = Geodude(ctx, package)
    if robot.pickup("can"):  # an instance, a type, or None for anything; either arm
        robot.place()  # into the bin on the holding arm's side, or "worktop"
    robot.right.pickup("can")  # an arm chosen by side
    robot.go_home()
```

- **Lookups:** `find_objects` and `holding`.
- **Moves:** `go_to` takes a configuration the package names. `go_home` goes to its `ready`, carrying whatever is held.
- **Per arm** (`robot.left` and `robot.right`): joint positions, the end-effector pose, the wrist wrench and its tare, and the gripper's `open` and `close`. `open` refuses while the hand holds an object.
- **Results:** every call returns a `TaskResult`. It is true when the call succeeded, and it records the object, the arm, whether the object is held and attached, and the skills' own results.
- **Abort:** `request_abort` cancels the active skill and leaves what is held as it is, reported so. Every call is refused until `clear_abort`.

The `recycling` functions beneath it remain available.

## Running it

From a clone, the application has its own [uv](https://docs.astral.sh/uv/) project and
environment in this directory:

```sh
git clone https://github.com/personalrobotics/geodude
cd geodude/v4
uv sync --locked
```

Recycle one can and record the run:

```sh
uv run geodude-recycling --seed 0 --record out/one
```

It prints the outcome, judged after the run from the can's resting pose, and where the
record was written:

```text
{"grasped": true, "dropped": true, "released": true, "landed": true, "detached": true, "in_bin": true}
recorded out/one/bundle and out/one/recycling.json
```

Recycle four cans, racing both arms and planning each can while the previous one is
carried:

```sh
uv run geodude-recycling --seed 4 --cans 4 --record out/four
```

```text
{"can_0": true, "can_1": true, "can_2": true, "can_3": true}
```

Sort a round of the legacy demo's mix:

```sh
uv run geodude-recycling --seed 0 --mix 5 --record out/mix
```

```text
{"cracker_box_0": "recycle_bin_0", "fuze_bottle_0": "recycle_bin_0", "gelatin_box_0": "recycle_bin_0", "pop_tarts_case_0": "recycle_bin_1", "sugar_box_0": "recycle_bin_1"}
```

`--to worktop` places the can upright on the worktop instead of dropping it, and
`--viser` serves a live view of the run at <http://localhost:8080>. The seed selects
the staged scene, and a deterministic run gives the same outcome for the same seed on
the same OpenBLAS kernel.

OpenBLAS picks its kernel for the CPU, and the kernels round differently, so one ulp of a
grasp region's orientation can change a round's outcome. CI checks the tests' outcomes
on the Haswell kernel, which every x86-64 host with AVX2 runs; select it with
`OPENBLAS_CORETYPE=Haswell`. `recycling.json` names the
kernel a run used (`blas_kernel`), and `null` when numpy does not use its bundled OpenBLAS, as on macOS
on Apple silicon.

## Geodude's choices

The regions offered to each skill are built in `src/geodude/regions.py` from each
object's prl_assets geometry. Grasps are sstsr's templates for the 2F-140, backed off
along the approach so that the skill can move straight in, and keep a box's fingertips
at least the planning clearance off the box. A drop releases the held object with its
lowest point just below the bin's rim, its footprint and, where they reach below the rim,
the open fingers inside the walls: released above the rim, an object strays sideways as
the pads let go, and can come to rest across it (#237). The region a dropped object must
come to rest in is the 3.x demo's, computed from the bin's geometry. These are Geodude's
choices rather than ssgrasping's, because another
robot or task would offer different grasps and containers.

The speeds, forces, and tolerances are set in `src/geodude/recycling.py`. Where a value
is set from measurement, its code comment records that evidence. Held cans are carried at
a fraction of the joint limits, because full speed can throw a can from the fingers. A
guarded placement stops at a contact force set above the free-motion peak. Free-space
moves keep a small clearance from the worktop, the bins, and the other cans. Straight
approaches allow a line width chosen from how far the arm actually arrives from its
planned line.

Control runs on what a physical robot has: joint, gripper and wrist channels, and the
registration of the objects on the table (#248). Contexts keep registered object poses
(ssrobot's `ObjectPoses.REGISTERED`), so planning and every decision use the poses last
registered, never the simulator's. Objects on the table are registered when a round
begins, before a race planned from a fresh snapshot once the scene has changed, and
after a recovery, before a retry. A grasp that closes on nothing (`missed`) may mean the
object moved: what was planned ahead is discarded, the table is registered again, and
the object is retried once. A dropped object, once its release is confirmed, is assumed to have landed and is
removed from the scene; a placed one is registered again where it rests.

The demo has no perception yet, so registration reads each object's privileged pose
channel, the simulator's ground truth, as the 3.x demo has always done, and the trace
records that source (`pose_stub`). Privileged state is otherwise read only to evaluate:
where each object came to rest, and how often an assumed landing was wrong.
`tests/test_registered_control.py` runs a round with every privileged channel withheld
and shows the same decisions, and moves a can behind registration's back: the grasp
misses, the table is registered again, and the retry succeeds.

## Scenes and evidence

`src/geodude/scene.py` stages each scene from prl_assets into its own directory. Object
poses are sampled with a seeded generator from sstsr's placement regions, and a
`manifest.json` records the prl_assets source, the seed, every placement, and the
SHA-256 of every staged file. The same inputs produce the same bytes.

`--record DIR` writes the run as an ssrobot run bundle in `DIR/bundle`, with the exact
robot, scene, and trace needed to inspect it elsewhere, and its outcome in
`DIR/recycling.json`. The outcome is a typed, versioned record,
`geodude.outcomes.RecyclingRun`. It holds:

- the seed and the OpenBLAS kernel;
- the name, version, and VCS commit of each distribution in the application's listed
  stack (`SOURCES` in `recycling.py`);
- the staged scene's manifest;
- the SHA-256 of the bundle's `bundle.json`;
- every attempt, with its race, its skill stages, and any failure and recovery;
- where each object came to rest.

It names no path, so a run directory can be moved and still verified: the bundle opens
on its own, and its `bundle.json` has the digest the outcome cites. Nothing in the
outcome depends on wall-clock time. A race records each candidate the chosen one
outranked without its plans, because how far those got depended on planning speed. The
same seed on the same kernel writes the same bytes.

The outcome is decoded strictly before it is written, and by every reader, with
ssrobot's codec. A missing, extra, or wrongly typed field is refused, as is an
inconsistent one, such as a drop released and evaluated without its landing. Each
refusal has its own code (`src/geodude/outcomes.py`), and nothing is coerced. The file
is replaced atomically, so a failed write never leaves a partial outcome.
`RecyclingRun.VERSION` versions the format. A bundle can be turned into a recording that anyone can
scrub with mj_viser, as described in its guide to [replaying a run
bundle](https://github.com/personalrobotics/mj_viser#replay-a-run-bundle).

## Current status

| Capability | Status |
| --- | --- |
| Pick one can and drop it into the bin on its side | Implemented; tested from the source tree and from the built wheel in a clean environment |
| Pick one can and place it upright on the worktop | Implemented; tested |
| Recycle several cans, racing both arms and planning each can during the previous transfer | Implemented; four cans tested, and the same round with planning delayed by up to 3 s gives the same trace (`test_planning_latency`) |
| Pick and drop each of the legacy demo's nine object types | Implemented; each type tested alone (`test_object_types`) |
| Sort a five-object round of the mix into the bins, recovering where it is safe | Implemented; seed 0 tested, its bundle verified and judged by the corpus evaluator (`test_sort_all`) |
| Recover a failed stage by its rule: release in place, detach a lost object, put back, retry | Implemented; a lift blocked by a shelf and a can pulled out mid-transfer each recover, judged against privileged poses (`test_recovered`); putting back (`test_unpickable`) and retrying (`test_registered_control`) |
| A robot API: pick up by instance, type or anything, place, go home, per-arm state and gripper, abort | Implemented; every call driven in one seeded scene and judged against privileged poses and the channels (`test_robot`) |
| Control on registered object poses | Implemented; a round with privileged channels withheld decides exactly as one with them (`test_registered_control`) |
| A typed, versioned, strictly decoded outcome | Implemented; tampered copies are each refused with their own code (`test_outcome_boundary`), and the outcome file is byte-identical with planning delayed (`test_planning_latency`) |
| Perception | Not implemented; registration reads privileged simulator channels |
| The physical robot | Not yet supported; it depends on ssrobot's hardware runtime |
| The 3.x application's other tasks | Still served by the application at the repository root |

Not every seed recycles every object. A run that fails reports the stage that decided
it, such as no arm finding a feasible prefix, and the record preserves the evidence
needed to diagnose it. `scripts/evaluate_recycling.py` measures reliability over fixed
corpora of rounds (personalrobotics/geodude#201): `development`, seeds 0-39, for
diagnosis and every fix; `catalog`, seeds 3000-3059, also for development, which deals
every catalog object that stages on the worktop (`recycling.CATALOG_KINDS`) ten times each
(#267); and two frozen corpora, each run once, whole, from a clean
tree: `acceptance`, seeds 1000-1039, #201's, and `cutover`, seeds 2000-2039, the Geodude
4.0 cutover's (#213), on the Haswell kernel. Both are spent. On the cutover corpus, 199
of 200 objects were recycled and 39 of 40 rounds cleared; its evaluation is
[`evidence/cutover/`](evidence/cutover/). On the development corpus, on the Haswell
kernel, 191 of 200 objects are recycled and 38 of 40 rounds cleared.

## Development

All commands run from `v4/`:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/check_boundary.py
uv run pytest
```

`scripts/check_boundary.py` confirms that Geodude 4.0 imports none of the 3.x stack:
mj_manipulator, mj_environment, asset_manager, or the old viewer. Each test writes its
evidence under `artifacts/` and documents how to reproduce it. CI also builds the
wheel, installs it in a clean environment, runs the demo from the installed command,
and checks with `scripts/check_installed.py` that the run used the installed 4.x
package rather than the source tree or the 3.x application.

CI runs each third-party action at a full commit, with its release in a comment.
Dependabot (`.github/dependabot.yml`) proposes each update as a pull request that moves
the pin and its comment together. Review it as any dependency change.
