# Test audit: ssgrasping skills and transitions (geodude#251)

Snapshot: ssgrasping `d29a4b4` (baseline `591f234`). Geodude reference: `geodude/v4` at `0618279`.
Read-only audit; nothing was run except one `gh run view` of ssgrasping's latest `main` CI
for step timings. Paths below are relative to the ssgrasping worktree unless prefixed.

## Scope and cost

| File | Scenarios | MuJoCo sessions | Notes |
| --- | --- | --- | --- |
| `tests/test_grasp_place.py` | 12 | 13 (`success` twice) | one test function, `:537` |
| `tests/test_drop.py` | 5 | 6 (`success` twice) | one test function, `:294` |
| `tests/test_lookahead.py` | 12 | 13 (`lookahead` twice) | one test function, `:605` |
| `tests/test_reference_robots.py` | 1 | 2 package loads, 10 steps each | `:42` |
| `scripts/study_transitions.py` + `evidence/transitions/` | 15 cases x 10 seeds (150 runs) | not in CI | evidence, not a gate |

Cost: the whole `uv run pytest` step took 85 s (py3.11), 105 s (py3.13), 131 s (py3.12) in
CI run 41369421956, so about 3 to 4 s per session on average. The per-test split is not in
the CI log (no `--durations`); `race` in `test_lookahead` is visibly the largest single
session (two races, two grasps, two drops, one landing). Every scenario is a fresh
`RobotContext` on a fresh `MujocoRuntime`; none shares a run with another gate today.
CI also runs the README's grasp example (3 to 5 s), which repeats `success`'s grasp.

Provenance: every scenario landed in the same squashed PR as the feature it proves
(`git log -S` per scenario name: `9d49ece`, `8d8cd07`, `69acd68`, `a9de6c0`, `64d82a7`,
`cec3d55`, `c1585e8`, `d29a4b4`). Whether a scenario was written before or after its
implementation cannot be recovered from squashed history; instead, each row below records
whether the scenario names an independent fault (an injected input or physical condition)
and an outcome derived from the contract, which is the policy's real concern.

## Cross-cutting checks

| Check | Protects | Covered elsewhere | Classification |
| --- | --- | --- | --- |
| `_reasons(scenarios)` (`test_grasp_place.py:495-523`, called `:572`, `test_drop.py:317`, `test_lookahead.py:675`) | Every failed stage carries a `Reason` and a cause in the vocabulary its reason names (ssrobot enums), never prose; successes carry none. Geodude's `safe_release` (`geodude/v4/src/geodude/recycling.py:849-861`) decides recovery from exactly (stage, reason, cause, held, attached). | Geodude `test_recovery.py` feeds fabricated `Stage`s, so it checks the consumer, not the producer. | Keep. Oracle is ssrobot's enums plus the README table (`README.md:60-76`), independent of ssgrasping code. `CAUSES` (`:482-490`) is a second copy of the README's cause column; acceptable as the oracle. |
| Repeat byte-identity: `first == again` (`test_grasp_place.py:545-546,571`, `test_drop.py:300-301,316`, `test_lookahead.py:609-610,674`) | Determinism on one host: manual clock (grasp+place), manual clock (grasp+drop+landing), lookahead with planning on workers (grasp+drop). | Geodude: `test_recycling.py:76` (grasp+drop+landing, whole trace sha256), `test_sort_all`, `test_registered_control` B twice, `test_planning_latency` (lookahead under randomized planner delay). | `test_lookahead`'s is the strongest in-repo (worker threads). `test_grasp_place`'s is the only determinism gate for `place` (Geodude's worktop run, `test_recycling.py:78`, runs once). `test_drop`'s is dominated: see D1. |
| `"environment"` recorded in every artifact | Byte-identity claimed within one environment | n/a | Keep; artifact metadata, not an assertion. |

## `tests/test_grasp_place.py`

| Scenario / assertion group | Protects (failure mode) | Covered elsewhere? | Change-detector / reimplementation? | Verdict |
| --- | --- | --- | --- | --- |
| `success` grasp stages and state (`:575-579`) | Full staged grasp succeeds; `acquired` claimed only after `lift` (#32). | Geodude `test_recycling` (grasp succeeds, application level), README example in CI. | Exact 12-name ordered list freezes incidental stages (`obstructed`, `clear`, `restrict`) beyond the semantic order (attach < lift < acquired). Stage names are public (README reason table; Geodude keys on `lift`, `transfer`), so partially contract. | Keep scenario. Assertion is a mild change-detector (C4). |
| `success` `retained` estimates and evaluations for lift and transfer (`:584-599`) | Carries estimate `retained` from the portable opening alone; attachment monitor only scores (#28, `c1585e8`). | Geodude `test_registered_control` (control without privileged channels) covers "evaluation steers nothing" at app level, not the estimate's content. Study `guarded.json` scored 98/0/0/20 at `ca18812`, not in CI. | `:591` re-does the guard comparison (`value >= below`): mild reimplementation, redundant with `outcome == confirmed`. Rest is semantic. | Keep. |
| `success` `placed` unobserved, `agrees is None` (`:600-611`) | No privileged pose decides `placed`; an unobserved estimate makes no claim (`d29a4b4`). | Drop's `landed` analogue (`test_drop.py:333-342,367-369`). Different skill, separate code path (`place.py:299-318` vs `drop.py:268-285`). | Semantic. | Keep. |
| `success` attachment transform kept through `restrict` (`:612-613`) | `restrict` re-declares the same transform; no silent re-attachment at a drifted pose. | No. | Semantic (declared vs carried). | Keep. |
| `success` place stages, attachment cleared (`:614-618`) | Place completes and detaches. | Geodude `test_recycling` worktop (`test_recycling.py:78,116-117`) with physical tolerances. | Exact stage list (C4). | Keep. |
| `success` tare rest timing (`:621-625`) | Descent (which tares) starts only after observed joint rest, not a fixed wait (#35, `cec3d55`). | No. | `:625` (`window[1] == end_ns`) checks ssrobot's `RestReport` shape: mild change-detector. `:624` equality to `allow_target`'s time is semantic (watch begins at transfer end). | Keep; `:625` optional to drop. |
| `success` gripper terminations (`:685-694`) | Each gripper stage waits on its command's own result (#29), close stops short on the can, release reaches target. | Study shows these terminal states across seeds (not in CI). | Semantic: these are the portable signals the acquired/missed/lost logic depends on. | Keep. |
| `tare_settled` (`:416-446`, `:629-632`) | Scientific invariant behind taring at rest: wrist force changes < 10 % of the guard threshold over 1 s after joint rest. | No. | Semantic, derived oracle (`TARE_SETTLED_FRACTION * CONTACT_N`). | Keep. Unique. Costs a full grasp + transfer + 1 s hold; cannot share `success`'s run because holding diverges the trajectory (ssrobot exposes no state fork). |
| `missed` (`:449-455`, `:634-636`) | Close reaching target means nothing in hand: fail `obstructed`/`missed`, nothing attached. | Geodude `test_registered_control.py:333` (moved can, `missed`, portable). Downstream only, and only on a pin bump. | Semantic. | Keep (cheap: ends at close; also the base of `obstructed`). |
| `obstructed` (`:638-649`) | A post the fingers stop on never yields `acquired`; the lift fails `not_planned/infeasible` with held+attached. This tuple is the producer side of Geodude's `RELEASE_IN_PLACE` (`recycling.py:859`). | Geodude `test_recovery.py:51-57` uses a hand-built copy of this tuple; no producer coverage downstream. | `:645-649` last-three-stage list partly duplicates `:644`; acceptable. | Keep. Unique boundary. |
| `lost` (`:650-653`) | 20 kg can: lift runs, re-close reaches target, `acquired` fails `lost`, portable evidence. | Study `lost` (7/10 seeds fail at the lift command instead; this seed reaches `acquired`). Not in CI. | Semantic. | Keep. Note seed sensitivity: the study shows most seeds fail earlier at the lift, so this assertion depends on the unperturbed start. |
| `preflight` (`:655-658`) | Ingress validation: an undeclared channel fails before anything is commanded (`time_ns == 0`). | No. | Semantic. | Keep. Cheap. |
| `external` (`:660-661`) | `run_skill` on an externally clocked runtime through `update()` alone. | No (Geodude uses the manual clock). | Asserts only state and last stage. | Keep. |
| `cancelled` (`:458-471`, `:663-667`) | Cancel mid-approach cancels the command, reports `cancelled`, attaches nothing. | No. | Semantic. | Keep. |
| `drifted` (`:669-674`) | `restrict` refuses to re-declare an attachment the monitor reports moved; kept attached, `held False`. | No. | Semantic. | Keep, but see Q1: this stage is decided on `privileged` evidence, which sits uneasily with "privileged state decides no stage" (`test_drop.py:26-29`). |
| `unconfirmed_release` (place) (`:676-679`) | Half-open release is not a release; object stays attached, `held None`. | Drop's `unconfirmed_release`. Release logic is copy-pasted in `place.py:291-305` and `drop.py:268-282`, so both are live. | Semantic. | Keep while the code is duplicated (see M3). |
| `trajectory_only` (`:282-300`, `:681`) | Neutral planning contract: a planner may return trajectories without paths. | No. | Asserts only success. | Keep (see M2, optional share). |
| `gripper_timeout` (`:696-699`) | A gripper deadline is a failure bound: close fails `command_failed/deadline_exceeded`. | No. | Semantic. | Keep. |

## `tests/test_drop.py`

| Scenario / assertion group | Protects | Covered elsewhere? | Change-detector / reimplementation? | Verdict |
| --- | --- | --- | --- | --- |
| `success` drop stages, detach (`:319-325`) | Drop releases over the bin and retreats. | Geodude `test_recycling.py:111` (drop stages, no retreat), `test_four_cans`, `test_sort_all`. Library path with retreat only here and in `registered`. | Exact stage list (C4). | Keep. |
| `success` landing (`:329-342`) | `landing` scores `landed` against the privileged pose for evaluation only; `agrees is None`. | Geodude `test_recycling.py:112` (`landing == [["observed", True]]`), `test_object_types`. | Semantic. | Keep. |
| `success` transitions (`:363-370`) | Transfer confirms `retained` (`opening_held`); `landed` unobserved (`no_portable_signal`). | Same contract as place (`test_grasp_place.py:584-611`), different skill code. | Reason strings are the estimate's machine vocabulary; semantic. | Keep. |
| `success` repeat run (`:301`, `:316`) | Determinism of grasp+drop+landing on a manual clock. | `test_lookahead` repeat (grasp+drop, plus workers), Geodude `test_recycling` repeat (grasp+drop+landing, trace sha256). | n/a | Deletion candidate D1. |
| `unconfirmed_release` (drop) (`:344-347`) | As place's, in `drop.py`. | Place's scenario covers the twin code. | Semantic. | Keep while duplicated (M3). |
| `registered` (`:349-360`) | Skills run with `ObjectPoses.REGISTERED`; the `released` policy removes the object so the retreat can plan; same stages, estimates, evaluations as `success` (differential oracle). | Geodude `test_registered_control` runs the whole demo registered, but its drops never retreat (`test_four_cans.py` docstring; `test_recycling.py:111`), so the release→removal→retreat path is unique here. | Differential against `success`; strong. | Keep. |
| `pulled` (`:233-254`, `:372-398`) | Opening guard stops a carry once the can leaves; `transition_refuted/retained`, `held False`, attached; stopped before the drop zone; gripper never opened. Producer of Geodude's `RELEASE_LOST` input (`recycling.py:861`). | Study `pulled`, `pulled_heavy` (10 seeds each, `guarded.json` at `ca18812`, not in CI). Geodude `test_recovery` uses a hand-built copy. | `:384` re-does the guard comparison (mild reimplementation, redundant with `outcome == refuted`). `:398` checks a `values` key's absence: object-shape check standing in for "no release stage" (C5). | Keep. Unique CI gate. |
| `monitor_only` (`:201-231`, `:257-271`, `:400-413`) | Privileged monitor never decides a stage: drop succeeds; evaluation records disagreement (`agrees False`). The "No privileged control" invariant (geodude#248). | Geodude `test_registered_control` B removes privileged channels entirely; it does not inject a disagreeing monitor. | Semantic. | Keep. Unique. |

## `tests/test_lookahead.py`

| Scenario / assertion group | Protects | Covered elsewhere? | Change-detector / reimplementation? | Verdict |
| --- | --- | --- | --- | --- |
| `plain` vs `lookahead` kinds (`:677-680`) | Lookahead changes no stage outcome or evidence kind (differential oracle). | Geodude `test_planning_latency` (planning time changes nothing) is related, not the same claim. | Strong differential oracle. | Keep. `plain` repeats `test_drop`'s non-lookahead grasp+drop in a different scene: see G1. |
| `lookahead` approach continued, lineage (`:681-683`) | Approach planned on the pregrasp's projection and continued, not replanned. | Geodude `test_four_cans.py:94-98` (lineage, `kept`) for raced grasps. | `:682` matches a prose `decision` string (C2). | Keep. |
| `lookahead` stage detail (`:684-685`) | Same as `:682`. | `:682`. | Exact `detail` prose (`_skill.py:649`). Change-detector; README says detail is prose and Geodude asserts detail must not drive decisions. | Delete assertion (C1). |
| `lookahead` boundaries `close, restrict, release` (`:686-692`) | Lookahead never crosses a stage whose outcome depends on an observation. | Geodude `test_four_cans.py:99` (drop: `["release"]`). | Exact list, but each entry is a semantic boundary. | Keep. |
| `lookahead` no log without lookahead (`:693`) | Lookahead is off by default. | No. | Key-presence check; weak but cheap. | Keep or drop; no run cost. |
| `transfer_ahead` (`:259-296`, `:695-710`) | Transfer planned once, on a worker, during the lift, on a prediction expecting `restrict`'s allowance; handed on and continued; a retreating drop refuses it (`invalid_spec`, `retreat_m`). | Geodude `test_four_cans` plans races during transfers, not a transfer during a lift. | `:708` exact detail prose (C1); `:705` prose decision (C2). `:710` thread check via `_Timed` is semantic (never on the context thread). | Keep scenario; drop `:707-708`. |
| `transfer_disturbed` (`:711-717`) | An unpredicted scene change between grasp and drop fails the handed-on transfer for recovery (`refused/unpredicted_scene_change`). | No. | `:717` prose decision (C2). | Keep. |
| `mismatch` (`:719-723`) | Arrival out of tolerance replans from the actual state and still succeeds. | No. | `:723` detail prose duplicates `:721` (C1). | Keep scenario; drop `:723`'s detail clause. |
| `scene_change` (`:359-375`, `:725-729`) | Unpredicted scene change during pregrasp fails the approach for recovery. | No. | `:729` prose decision (C2). | Keep. |
| `race`, first race (`:378-419`, `:731-759`) | Nothing moves before selection; preferred-but-unreachable left arm fails at `transfer`, every transfer seed tried from one lift prefix without replanning the prefix; `plan_status` recorded per plan. | Geodude `test_four_cans` asserts selection-by-preference only when the preferred succeeds; it never exercises a preferred arm failing at transfer, seed reuse, or `plan_status`. | `:740-753` inspects `chains` structure (stage seeds, attempts): semantic (no prefix replanning), tied to ssrobot's chain report shape. `:412-418` carry a diagnostic for "an intermittent CI failure" of `:732`: a known flake, uninvestigated here (Q2). | Keep. |
| `race`, `race_b` on prediction, pregrasp during transfer, approach kept (`:761-764`, `:776-787`) | Next race planned on the transfer's prediction with `can` released; pregrasp continued after release; raced approaches not replanned. | Largely Geodude `test_four_cans.py:83-110` (basis `prediction`, `pregrasp.during == transfer`, `kept`, decisions after the transfer ends) and `test_planning_latency`. Downstream, on pin bump only. | `:776-778` recomputes the selection rule (`min` by `-priority, cost_s`), a reimplementation of ssrobot/ssgrasping's rule (`candidates.py:34-35`) that also omits its declaration-order tie-break; the concrete `== "right"` already carries the claim (C6). | Keep the group (library proof of `Candidates(after=..., releases=...)`); delete `:776-777`'s rule recomputation. |
| `race`, landing alongside `grasp_b` (`:770-775`) | `can_b`'s grasp starts at the release; `can`'s landing is observed beside it. | Geodude `test_recycling` landing, not concurrent. | `:775` freezes `land_ns`, which `landing.py:11` calls "a temporary fixed wait": change-detector against the planned event-driven landing (C3). | Keep; rewrite `:775` as an ordering (`landed.time_ns > grasp_b start`). |
| `race`, `drop_b` (`:483-487`, `:789`) | `can_b` dropped after the raced grasp. | Every drop scenario; Geodude `test_four_cans`. | n/a | Trim candidate M4 (saves one transfer). |
| `not_at_rest`, `stale_rest` (`:299-356`, `:794-810`) | Continuation is refused `not_at_rest` once when the arm is moving or readings predate the arrival; nothing submitted after the refusal (trace order). | No. | `:810` matches detail prose; the structured part (`rest.outcome == at_rest`, `waited < timeout`, `:808-809`) does not by itself show staleness, so the prose is the only oracle for "stale" (C1, needs a structured field). | Keep both. |
| `deepest` (`:553-571`, `:815-819`) | Race checks the guaranteed `approach_m`; the grasp approaches as deep as plans, short of the blocked line. | No. | Semantic bounds. | Keep. |
| `closed` (`:574-592`, `:823-825`) | A closed race reports `canceled`/`aborted`, not `canceled_by_priority`. | No. | Semantic. | Keep. Cheap (no motion). |
| `clearance` (`:502-545`, `:829-832`) | Clearance applies only to free-space stages on every path (raced pregrasp, transfer 5 mm; continued approach, lift 0). | No. | `:832` exact ordered list of compiled clearances also freezes that exactly four commands are compiled in that order; semantic content is per-stage. | Keep. |
| `timing.json` (`:656-672`) | Nothing asserted (outside the claim) except `:709-710`. | n/a | Artifact only. | Keep. |

## `tests/test_reference_robots.py`

| Assertion | Protects | Covered elsewhere? | Classification |
| --- | --- | --- | --- |
| Both packages installed (`:46`) | The test environment uses installed packages, not checkouts. | Every Geodude scenario loads `geodude_assets` the same way (`test_grasp_place.py:368`). ADA: only here in ssgrasping. | Geodude half redundant. |
| `time_ns == STEPS * period` (`:47`) | ssrobot's manual-clock stepping. | ssrobot's own tests (owner; not verified by me). | Wrong owner for ssgrasping. |
| 4 Geodude manipulators, ADA `["arm"]` (`:48-49`) | Nothing of ssgrasping's; robot package content. | n/a | Change-detector. |

It backs one README claim (`README.md:372`: "the ADA package opens in the test
environment") and is the only consumer of the `ada-assets` dev dependency
(`pyproject.toml:27`). See D2.

## The study as evidence (`scripts/study_transitions.py`, `evidence/transitions/`)

- Role: scientific basis for the transition design (which portable signals decide which
  transition; the opening guard's 0.441 separation; `placed`/`landed` unobservable). It is
  not a gate: not in CI, 150 runs per grid on `siddh`. `summary.json` was recorded at
  `bbb13c0`, `guarded.json` at `ca18812`, both clean and older than `d29a4b4`; the README
  names the commits, so the evidence is honestly dated, but neither reflects `d29a4b4`'s
  `placed`/`landed` estimates or `agrees: None`.
- Relation to tests: the tests are single-seed CI gates on the same faults (`missed`,
  `obstructed`, `lost`, `pulled`, partial open). The study cannot replace them (not in CI,
  no assertions); they cannot replace the study (one seed, no truth recording). No
  overlap to remove. Cases with no test counterpart (`thrown`, `stuck`, `outside_bin`,
  `outside_region`, `_heavy`) are deliberately study-only: they establish what is
  unobservable, and the tests assert the resulting `unobserved` estimates.
- Coupling: the study imports private test helpers and constants
  (`study_transitions.py:51-67`: `_grasp_spec`, `_place_spec`, `_planners`, `_report`,
  `_drop_spec`, `_scene`, ...), as do `test_drop.py:65-81` and `test_lookahead.py:108-123`.
  Any test deletion or merge must keep these or move them to a shared fixtures module
  first; otherwise the committed evidence becomes unreproducible.
- `CONTACT_N` (`test_grasp_place.py:121-131`) and `CARRY_SCALING` (`:137-143`) carry their
  measurement history in comments. These are derived values with evidence, not tuned
  constants, but the measurements live only in prose: no artifact reproduces them.

## (1) Keep: uniquely protects something

29 of 30 scenarios. Unique in the whole stack (no ssgrasping or Geodude coverage):
`tare_settled`, `obstructed` (producer of Geodude's in-place release), `lost`, `preflight`,
`external`, `cancelled`, `drifted`, `trajectory_only`, `gripper_timeout`, `pulled` (producer
of Geodude's lost-object release), `monitor_only`, `registered` (release→remove→retreat),
`plain`/`lookahead` (differential), `transfer_ahead`, `transfer_disturbed`, `mismatch`,
`scene_change`, `not_at_rest`, `stale_rest`, `deepest`, `closed`, `clearance`, the first
`race`. Covered downstream but kept because Geodude sees ssgrasping only on a pin bump and
the layering rule requires the library to prove its own skills: `success` (place and drop),
`missed`, `race`'s `race_b` tail. Both `unconfirmed_release` scenarios stay while
`place.py` and `drop.py` duplicate the release code.

## (2) Deletion and merge candidates

| ID | Candidate | Real failure mode considered | Replacing coverage | Unique boundary lost | Confidence |
| --- | --- | --- | --- | --- | --- |
| D1 | `test_drop.py`'s repeat run (`:301`, `:316`, `"repeat_identical"`) | Nondeterminism introduced in `drop` or `landing` on a manual clock. | `test_lookahead.py:609-610,674` repeats grasp+drop (same `drop` code, plus workers). Geodude `test_recycling.py:76,118` repeats grasp+drop+landing and compares the trace sha256. | Drop-with-retreat determinism (`lookahead` drops retreat too, `:246-253`, so covered); landing determinism only downstream. Saves one session. | Medium |
| D2 | `test_reference_robots.py` as a whole, together with the README claim and the `ada-assets` dev dependency | ADA's package cannot be loaded in ssgrasping's locked environment. | None today. The ADA cylinder slice (next in the sequence) would replace it with a real ADA skill scenario. | "ADA opens here" until that slice lands. | Retire when the ADA slice lands; do not delete standalone. |
| C1 | Detail-prose asserts: `test_lookahead.py:685`, `:708`, `:723` (detail clause), `:810` | A wording change fails CI with no behavior change. | `:682`, `:706`, `:721` assert the same decision from the lookahead log. `:810` has no structured replacement. | None for `:685/:708/:723`. `:810` is the only oracle that a stale rest was stale; it needs a structured field in the log (production follow-up) before removal. | High (first three) |
| C3 | `test_lookahead.py:775` (`landing time == released + land_ns`) | Freezes the temporary fixed landing wait (`landing.py:11`); breaks when landing becomes event-driven. | Rewrite as an ordering: landing after `grasp_b`'s first stage. | None. | High |
| C6 | `test_lookahead.py:776-777` (selection-rule recomputation) | Reimplements the race's selection rule, incompletely (no declaration-order tie-break). | `:778` `== "right"`; ssrobot's race tests own the rule (other auditor). | None. | Medium |
| C5 | `test_drop.py:398` (`"released_opening" not in values`) | Object-shape stand-in for "the gripper never opened". | Replace with "no `release` stage" in `pulled["stages"]`; `:375` already ends at `transfer`. | None. | Medium |
| C8 | `test_reference_robots.py:48-49` | Robot package content change-detector. | Goes with D2. | None. | High |
| M4 | `race`'s `drop_b` (`test_lookahead.py:483-487`, `:789`) | `can_b`'s drop after a raced, continued grasp fails. | Every drop scenario; Geodude `test_four_cans` drops each raced can. | None identified. Saves one transfer. | Low (small saving) |

Considered and not recommended:
- Merging `external` with `trajectory_only` (M2) saves one session (~3 s) but couples two
  unrelated contract paths in one failure; not worth it.
- Deleting `missed` or the `race_b` tail because Geodude covers them: Geodude pins
  ssgrasping, so a regression surfaces only on the next pin bump, in another repo.
- Rewriting exact stage lists (C4: `test_grasp_place.py:576-579,615-617`,
  `test_drop.py:322-324`, `test_lookahead.py:765-767`): stage names are public (README reason
  table; Geodude keys on `lift`, `transfer`) and the retreat-free list at `:765-767` is the
  semantic claim of `retreat_m = 0`. These are mild change-detectors; rewriting them is
  low value.
- Prose `decision` strings in the lookahead log (C2: `:682`, `:705-706`, `:717`, `:721`,
  `:729`, `:786`, `:799`; also Geodude `test_four_cans.py:98,110`): semantic but encoded
  as prose (`_skill.py:364,385,619,649-659`). The fix is a structured decision kind and
  refusal code in production (a follow-up), not test deletion.

## (3) Expensive runs to generate once and check several times

- G1: `test_drop`'s `success` and `test_lookahead`'s `plain` are the same non-lookahead
  grasp+drop of `can` into the right bin, in two scenes (`test_lookahead` adds `can_b` and
  a left bin). Running `test_drop` in `test_lookahead`'s scene would let one run serve
  both. Saves one session. It would make `drop.json` depend on a richer scene; low value.
- G2: three repeat-identity runs (`test_grasp_place`, `test_drop`, `test_lookahead`) each
  duplicate a scenario. Keep `test_grasp_place`'s (the only `place` determinism gate) and
  `test_lookahead`'s (workers); D1 removes `test_drop`'s.
- G3: `tare_settled` re-runs `success`'s grasp and transfer to the rest tick. It cannot
  share the run without a state fork, which ssrobot does not expose. No action.
- G4: the README example in CI repeats `success`'s grasp. It is the doc's owner-check; keep.
- Total possible saving from D1, G1, M4: about 3 sessions of 34, roughly 10 s of an 85 to
  131 s step. The suite is already cheap per session; the audit's value here is in the
  change-detector asserts, not in run count.

## Open questions for the consolidator

- Q1: `drifted` fails `restrict` with `ATTACHMENT_LOST` on `privileged` evidence
  (`test_grasp_place.py:671-673`, `_skill.py:798`), so a privileged monitor decides a
  stage. `monitor_only` (`test_drop.py:400-413`) asserts the monitor never decides one.
  The two may be consistent (the first is ssrobot refusing a re-declaration, not the skill
  reading truth), but on a physical robot without a monitor `restrict` would never fail
  this way. Worth checking against the "No privileged control" rule (geodude#248). Not
  resolved here.
- Q2: `test_lookahead.py:412-418` records diagnostics for "an intermittent CI failure" of
  `nothing_moved_before_selection` (`:732`). A known flake in the largest session; I did not
  find its issue or root cause.
- Q3: test helpers shared by import across test modules and the study
  (`study_transitions.py:51-67`) are accidental complexity; a fixtures module would make
  deletions safe. This is a refactor, not part of the test audit.
