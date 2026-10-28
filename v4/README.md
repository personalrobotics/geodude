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
can to be released over the bin and to come to rest on its floor, and the selected
pregrasp continues once the release has happened. The demo can also place a single can
upright at a target on the worktop under a force-guarded descent, instead of dropping
it.

The legacy demo's round stages five objects drawn, by the seed, from its nine types:
`can`, `spam_can`, `cracker_box`, `gelatin_box`, `sugar_box`, `pocky_box`,
`pop_tarts_case`, `fuze_bottle` and `plastic_glass` (`recycling.MIX_KINDS`).
`recycling.sort_all` sorts them all into the bins, racing both arms as above. An attempt
that fails is recovered only where releasing the object is safe: a lift that did not
plan releases it where it was grasped, and an object that slipped in the carry is
released over the bin. An object released back onto the worktop upright is tried once
more. Anything else stops the round, the object still declared held. The table is
clear once every object rests in a bin.

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
the staged scene, and a deterministic run gives the same outcome for the same seed.

## Geodude's choices

The regions offered to each skill are built in `src/geodude/regions.py` from each
object's prl_assets geometry. Grasps are sstsr's templates for the 2F-140, backed off
along the approach so that the skill can move straight in. The drop zone over a bin and
the region a dropped can must come to rest in are the 3.x demo's, computed from the
bin's geometry. These are Geodude's choices rather than ssgrasping's, because another
robot or task would offer different grasps and containers.

The speeds, forces, and tolerances are set in `src/geodude/recycling.py`. Where a value
is set from measurement, its code comment records that evidence. Held cans are carried at
a fraction of the joint limits, because full speed can throw a can from the fingers. A
guarded placement stops at a contact force set above the free-motion peak. Free-space
moves keep a small clearance from the worktop, the bins, and the other cans. Straight
approaches allow a line width chosen from how far the arm actually arrives from its
planned line.

The demo has no perception. Each object's pose is read from its privileged pose
channel, the simulator's ground truth, as the 3.x demo has always done. ssrobot marks
those channels as privileged, so the record distinguishes them from the observations a
physical robot would provide.

## Scenes and evidence

`src/geodude/scene.py` stages each scene from prl_assets into its own directory. Object
poses are sampled with a seeded generator from sstsr's placement regions, and a
`manifest.json` records the prl_assets source, the seed, every placement, and the
SHA-256 of every staged file. The same inputs produce the same bytes.

`--record DIR` writes the run as an ssrobot run bundle in `DIR/bundle`, with the exact
robot, scene, and trace needed to inspect it elsewhere, and its outcome in
`DIR/recycling.json`. The outcome records the version and origin of each distribution in
the application's listed stack (`SOURCES` in `recycling.py`), every skill stage, and
where each can came to rest. A bundle can be turned into a recording that anyone can
scrub with mj_viser, as described in its guide to [replaying a run
bundle](https://github.com/personalrobotics/mj_viser#replay-a-run-bundle).

## Current status

| Capability | Status |
| --- | --- |
| Pick one can and drop it into the bin on its side | Implemented; tested from the source tree and from the built wheel in a clean environment |
| Pick one can and place it upright on the worktop | Implemented; tested |
| Recycle several cans, racing both arms and planning each can during the previous transfer | Implemented; four cans tested |
| Pick and drop each of the legacy demo's nine object types | Implemented; each type tested alone (`test_object_types`) |
| Sort a five-object round of the mix into the bins, recovering where it is safe | Implemented; seed 0 tested twice, its bundles verified (`test_sort_all`) |
| Perception | Not implemented; object poses come from privileged simulator channels |
| The physical robot | Not yet supported; it depends on ssrobot's hardware runtime |
| The 3.x application's other tasks | Still served by the application at the repository root |

Not every seed recycles every object. A run that fails reports the stage that decided
it, such as no arm finding a feasible prefix, and the record preserves the evidence
needed to diagnose it. `scripts/evaluate_recycling.py` measures reliability over fixed
corpora of rounds (personalrobotics/geodude#201): `development`, seeds 0-39, for
diagnosis and every fix, and `acceptance`, seeds 1000-1039, run once, whole, from a
clean tree. On the development corpus, 196 of 200 objects are recycled and 37 of 40
rounds cleared.

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
