# Trajectory Following Issue - Investigation & Fix

## Problem Summary

After adding closed-loop trajectory following, bimanual demos show trajectory tracking failures with large position errors (3-46°) at trajectory start, even though the debug single-arm demo works perfectly.

## Root Cause #1: Planner State Corruption (FIXED)

**Issue**: The motion planner (`plan_to_configuration()`) corrupts the robot state during collision checking and RRT exploration. After planning completes, the robot's `qpos` is left at some random explored configuration, not at the start position where the trajectory expects to begin.

**Location**: `geodude/src/geodude/arm.py`, line 633
```python
path = planner.plan(q_start, goal_tsrs=[goal_tsr], seed=seed)
# After this, robot state is corrupted!
return path  # Returned without restoring state
```

**Fix Applied**: Added state restoration after planning (line 642-646):
```python
# CRITICAL: Restore robot state after planning
# Planner corrupts state during collision checking and exploration
self.set_joint_positions(q_start)
for i in range(len(self.data.qvel)):
    self.data.qvel[i] = 0.0
```

**Result**: Discontinuity warnings eliminated! Planner now returns with robot at `q_start` (path[0]).

## Root Cause #2: Weak PID Controller Gains (FIXED)

**Issue**: Default PID gains (kp=25, kd=0) were too weak to handle large initial tracking errors from actuator lag. The controller couldn't recover from 3-4° initial errors, leading to catastrophic failures (3° → 23° → 46°).

**Symptoms**:
- Right arm: 3.94° error at waypoint 20, growing to 23° by waypoint 120, catastrophic 46° by waypoint 1067
- Left arm: 0.01° error throughout (happens to be close to start position already)
- Controller unable to correct errors, leading to divergence

**Fix Applied**: Tuned PID gains in `geodude/src/geodude/config.py` and `geodude/src/geodude/executor.py`:
```python
kp: float = 100.0  # Was 25.0 - now aggressive position correction
ki: float = 0.0    # Keep at 0 for trajectory tracking (no steady-state error accumulation)
kd: float = 10.0   # Was 0.0 - now provides damping and velocity tracking
```

**Result**: Dramatically improved tracking! Controller now recovers from initial errors:
- Right arm: 3.95° → 12.75° → 0.97° → 0.4° (successful recovery within 60 waypoints)
- Left arm: 0.21-0.38° throughout (excellent)
- Most trajectories complete successfully with <1° average error

**Why these gains**:
- **kp=100**: Aggressive position correction for robust tracking
- **ki=0**: Not needed for trajectory tracking (can cause overshoot and oscillation)
- **kd=10**: Velocity feedback for damping and smooth motion

## Root Cause #3: Teleportation During Goal Pose Computation (FIXED)

**Issue**: When planning a new trajectory, the code teleported the robot to the goal configuration to compute the end-effector pose, then teleported back. This was visible in the viewer as a "huge jump" between trajectories.

**Location**: `geodude/src/geodude/arm.py`, lines 612-615 (old code)
```python
# Get the EE pose at the goal configuration
old_q = self.get_joint_positions()
self.set_joint_positions(q_goal)  # ← Visible teleportation!
goal_pose = self.get_ee_pose()
self.set_joint_positions(old_q)   # ← Teleport back
```

**Symptoms**:
- "When the right arm finishes its first trajectory, it teleports to the side of the workspace and runs its next trajectory"
- Large visible discontinuities between consecutive trajectories
- Happens at the START of planning, not during execution

**Fix Applied**: Created `_get_ee_pose_at_config()` method that computes forward kinematics using a temporary MjData copy:
```python
def _get_ee_pose_at_config(self, q: np.ndarray) -> np.ndarray:
    """Compute end-effector pose at a specific configuration without modifying robot state.

    Creates a temporary MjData copy to avoid corrupting the shared robot state.
    This is essential during planning to avoid visible teleportation artifacts.
    """
    temp_data = mujoco.MjData(self.model)
    temp_data.qpos[:] = self.data.qpos
    temp_data.qvel[:] = self.data.qvel

    # Set arm joints to target configuration in temp data
    for i, qpos_idx in enumerate(self.joint_qpos_indices):
        temp_data.qpos[qpos_idx] = q[i]

    # Compute FK in temp data (doesn't affect shared state)
    mujoco.mj_forward(self.model, temp_data)

    # Read EE pose from temp data
    pos = temp_data.site_xpos[self.ee_site_id]
    rot_mat = temp_data.site_xmat[self.ee_site_id].reshape(3, 3)

    transform = np.eye(4)
    transform[:3, :3] = rot_mat
    transform[:3, 3] = pos
    return transform
```

Updated `plan_to_configuration()` to use the new method:
```python
# Get the EE pose at the goal configuration WITHOUT modifying robot state
# This avoids visible teleportation artifacts in the viewer
goal_pose = self._get_ee_pose_at_config(q_goal)
```

**Result**: No more visible teleportation between trajectories! Smooth transitions when chaining multiple movements.

## Files Modified

1. **`geodude/src/geodude/arm.py`** (line 642-646)
   - Added state restoration after `planner.plan()`

2. **`geodude/examples/basic_movement.py`**
   - Simplified settlement code (removed complex loops)
   - Added actuator command initialization
   - Added success/failure tracking

## Diagnostic Commands

```bash
cd geodude
.venv/bin/mjpython examples/basic_movement.py
# Watch for:
# - "WARNING: Unexpected discontinuity" (should not appear if planner fix works)
# - Initial tracking errors at waypoint 20
# - "ABORT: Catastrophic tracking error" messages
```

## Root Cause #4: Collision-Induced Tracking Failures (IDENTIFIED)

**Issue**: Catastrophic tracking errors (16° → 46° ABORT) are caused by physical collisions during trajectory execution, not weak controller gains. When the arm collides with the base or self-collides, physics prevents motion while the controller keeps commanding large corrections, creating huge position errors.

**Symptoms**:
- Tracking errors suddenly spike: 2.09° → 16.19° → 46.2° ABORT
- Errors coincide with physical contacts
- Doubling gains (kp=100→200, kd=10→20) has no effect on failure rate

**Diagnosis**: Added collision detection during trajectory execution:
```python
if robot.data.ncon > 0:
    # Check contacts for arm-base collisions
    for contact_idx in range(robot.data.ncon):
        contact = robot.data.contact[contact_idx]
        # ... check if moving arm hits base or other obstacles
        if collision_detected:
            print(f"ABORT: Collision detected between {body1_name} and {body2_name}")
            return False
```

**Result**: Successfully detected collision at waypoint 373: `"Collision detected between vention_base and right_ur5e/gripper/right_follower"`

**Why this happens**:
1. Planner checks discrete waypoints for collisions
2. Actuator dynamics cause deviations between waypoints
3. Small tracking errors (0.5-2°) accumulate
4. Arm slightly overshoots and makes unexpected contact
5. Collision prevents motion → huge position error → ABORT

**Proposed Solutions**:
1. **Add safety margin to collision checker** - inflate geometry slightly during planning
2. **Increase collision checking resolution** - check more intermediate points
3. **Monitor contacts during execution** - abort gracefully on collision (currently implemented)
4. **Reduce trajectory aggressiveness** - lower velocity/acceleration limits for safer tracking

---
*Document created: 2026-01-22*
*Last updated: 2026-01-22*
*Status: Root causes 1-3 fixed, Root cause 4 identified (collision handling needed)*
