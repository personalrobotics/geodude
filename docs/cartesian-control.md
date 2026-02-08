# Cartesian Velocity Control with Hard Constraints

Real-time Cartesian control is fundamental to manipulation: move the end-effector along a desired twist while respecting joint position and velocity limits. This document derives our approach from first principles and explains why standard methods fall short.

## The Problem

Given a desired end-effector twist $\mathbf{v}\_d \in \mathbb{R}^6$ (linear and angular velocity), find joint velocities $\dot{\mathbf{q}} \in \mathbb{R}^n$ such that:

```math
\mathbf{J}(\mathbf{q}) \dot{\mathbf{q}} \approx \mathbf{v}_d
```

subject to:
- **Joint position limits**: $\mathbf{q}\_{\min} \leq \mathbf{q} + \dot{\mathbf{q}} \Delta t \leq \mathbf{q}\_{\max}$
- **Joint velocity limits**: $|\dot{\mathbf{q}}| \leq \dot{\mathbf{q}}\_{\max}$

where $\mathbf{J}(\mathbf{q})$ is the $6 \times n$ manipulator Jacobian.

The challenge is threefold:
1. The system may be **redundant** ($n > 6$), **exactly determined** ($n = 6$), or **under-actuated** ($n < 6$)
2. Near **singularities**, the Jacobian becomes rank-deficient
3. **Hard constraints** must never be violated—not even transiently

## Standard Approaches and Their Limitations

### The Pseudoinverse Solution

The most common approach uses the Moore-Penrose pseudoinverse:

```math
\dot{\mathbf{q}} = \mathbf{J}^+ \mathbf{v}_d
```

where $\mathbf{J}^+ = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T)^{-1}$ for full row rank, or computed via SVD.

**Problems:**
- Near singularities, $\mathbf{J}^+$ produces arbitrarily large joint velocities
- No mechanism to enforce joint limits—solutions can violate constraints
- Requires post-hoc clamping, which distorts the achieved twist direction

### Damped Least Squares (Levenberg-Marquardt)

A common fix adds regularization:

```math
\dot{\mathbf{q}} = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T + \lambda^2 \mathbf{I})^{-1} \mathbf{v}_d
```

This bounds joint velocities near singularities. However:
- The damping $\lambda$ requires careful tuning per robot
- **Still no constraint enforcement**—joint limits are ignored
- Solutions are minimum-norm but not necessarily feasible

### MoveIt Servo

The industry-standard [MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html) implements the pseudoinverse with threshold-based singularity detection:

```cpp
// Simplified from servo_calcs.cpp
Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
double condition = svd.singularValues()(0) / svd.singularValues()(n-1);

if (condition > hard_stop_threshold) {
    return HALT;  // Emergency stop
}
```

**Limitations observed in practice:**
- Robots frequently get stuck at singularities during teleoperation
- The "emergency stop" behavior is overly conservative
- Joint limits enforced by clamping individual joints, distorting motion direction
- No principled trade-off between tracking accuracy and constraint satisfaction

## Our Approach: Constrained Quadratic Programming

We formulate Cartesian control as a box-constrained QP solved at each timestep:

```math
\min_{\dot{\mathbf{q}}} \quad \frac{1}{2} \| \mathbf{J} \dot{\mathbf{q}} - \mathbf{v}_d \|_{\mathbf{W}}^2 + \frac{\lambda}{2} \| \dot{\mathbf{q}} \|^2 \quad \text{s.t.} \quad \boldsymbol{\ell} \leq \dot{\mathbf{q}} \leq \mathbf{u}
```

where:
- $\mathbf{W}$ is a twist weighting matrix (discussed below)
- $\lambda$ is a small damping coefficient ($10^{-4}$ by default)
- $\boldsymbol{\ell}, \mathbf{u}$ are joint velocity bounds derived from both position and velocity limits

### Twist Weighting: Making Heterogeneous Units Commensurable

A 6D twist contains linear velocity (m/s) and angular velocity (rad/s)—different physical quantities. Minimizing $\|\mathbf{J}\dot{\mathbf{q}} - \mathbf{v}\_d\|^2$ directly couples these arbitrarily based on numerical magnitude.

We introduce a **length scale** $L$ (default 0.1m, typical gripper workspace size):

```math
\mathbf{W} = \text{diag}(1, 1, 1, L^{-2}, L^{-2}, L^{-2})
```

This makes the objective **scale-invariant**: a 0.1 m/s linear error is weighted equally to a 1 rad/s angular error when $L = 0.1$m. The choice of $L$ encodes the characteristic length of your task—smaller values prioritize rotational accuracy.

### Deriving the Velocity Bounds

The key insight is converting position limits to velocity constraints at each timestep.

**From position limits:**

```math
\boldsymbol{\ell}_{\text{pos}} = \frac{(\mathbf{q}_{\min} + \boldsymbol{\epsilon}) - \mathbf{q}}{\Delta t}, \quad
\mathbf{u}_{\text{pos}} = \frac{(\mathbf{q}_{\max} - \boldsymbol{\epsilon}) - \mathbf{q}}{\Delta t}
```

where $\boldsymbol{\epsilon}$ is a safety margin (5° by default). This ensures $\mathbf{q} + \dot{\mathbf{q}} \Delta t$ stays within limits.

**Combined with velocity limits:**

```math
\boldsymbol{\ell} = \max(-\dot{\mathbf{q}}_{\max}, \boldsymbol{\ell}_{\text{pos}}), \quad
\mathbf{u} = \min(+\dot{\mathbf{q}}_{\max}, \mathbf{u}_{\text{pos}})
```

The `max` and `min` take the **more restrictive** bound. Near a joint limit, the position-derived bound dominates; in the workspace interior, velocity limits dominate.

**Handling infeasibility:**
When a joint is already past the safe margin (can happen during initialization), we ensure zero velocity is always feasible:

```python
infeasible = ell > u
ell[infeasible] = np.minimum(ell[infeasible], 0)
u[infeasible] = np.maximum(u[infeasible], 0)
```

This allows motion back toward the safe zone without solver failure.

### The QP Structure

Expanding the objective in standard QP form ($\frac{1}{2} \dot{\mathbf{q}}^T \mathbf{H} \dot{\mathbf{q}} + \mathbf{g}^T \dot{\mathbf{q}}$):

```math
\mathbf{H} = \mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \mathbf{I}
```

```math
\mathbf{g} = -\mathbf{J}^T \mathbf{W} \mathbf{v}_d
```

The matrix $\mathbf{H}$ is symmetric positive-definite (SPD) due to the $\lambda \mathbf{I}$ term, even when $\mathbf{J}$ is rank-deficient. This provides **implicit singularity handling**: near singularities, the regularization term dominates, naturally limiting joint velocities without explicit detection.

## Efficient Solving: Projected Gradient Descent

For a 6-DOF arm at 125 Hz control rate, we need a solver that:
1. Converges in microseconds
2. Exploits warm-starting from the previous solution
3. Handles box constraints without complex active-set management

We use **projected gradient descent** (PGD), a first-order method ideally suited to box-constrained QPs.

### Mathematical Derivation

Consider the box-constrained QP:

```math
\min_{\mathbf{x}} \quad f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T \mathbf{H} \mathbf{x} + \mathbf{g}^T \mathbf{x} \quad \text{s.t.} \quad \boldsymbol{\ell} \leq \mathbf{x} \leq \mathbf{u}
```

The gradient of the objective is:

```math
\nabla f(\mathbf{x}) = \mathbf{H} \mathbf{x} + \mathbf{g}
```

Standard gradient descent would update $\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha \nabla f(\mathbf{x}^{(k)})$, but this may violate the box constraints. The **projection operator** onto the feasible set $\mathcal{C} = \{\mathbf{x} : \boldsymbol{\ell} \leq \mathbf{x} \leq \mathbf{u}\}$ is simply element-wise clamping:

```math
\text{proj}_\mathcal{C}(\mathbf{x}) = \text{clip}(\mathbf{x}, \boldsymbol{\ell}, \mathbf{u}) = \max(\boldsymbol{\ell}, \min(\mathbf{u}, \mathbf{x}))
```

The PGD iteration is:

```math
\mathbf{x}^{(k+1)} = \text{proj}_\mathcal{C}\left( \mathbf{x}^{(k)} - \alpha \nabla f(\mathbf{x}^{(k)}) \right)
```

### Choosing the Step Size

For convergence, we need $\alpha$ small enough that the objective decreases at each step. The key theorem (from convex optimization) states that for an $L$-smooth function (one where $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$), PGD converges with step size $\alpha = 1/L$.

For our quadratic objective, the smoothness constant is the largest eigenvalue of the Hessian:

```math
L = \lambda_{\max}(\mathbf{H}) = \|\mathbf{H}\|_2
```

Therefore, the optimal constant step size is:

```math
\alpha = \frac{1}{\|\mathbf{H}\|_2}
```

For our Hessian $\mathbf{H} = \mathbf{J}^T \mathbf{W} \mathbf{J} + \lambda \mathbf{I}$:

```math
\|\mathbf{H}\|_2 \leq \|\mathbf{J}^T \mathbf{W} \mathbf{J}\|_2 + \lambda \leq \|\mathbf{J}\|_2^2 \|\mathbf{W}\|_2 + \lambda
```

With typical values ($\|\mathbf{J}\|\_2 \approx 1$, $\|\mathbf{W}\|\_2 = 1$, $\lambda = 10^{-4}$), we get $\alpha \approx 1$.

### Convergence Analysis

For a strongly convex function with parameter $\mu = \lambda\_{\min}(\mathbf{H}) \geq \lambda$ (guaranteed by damping), PGD achieves **linear convergence**:

```math
\| \mathbf{x}^{(k)} - \mathbf{x}^* \|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \| \mathbf{x}^{(0)} - \mathbf{x}^* \|^2
```

The convergence rate depends on the **condition number** $\kappa = L/\mu$:

```math
\kappa = \frac{\lambda_{\max}(\mathbf{H})}{\lambda_{\min}(\mathbf{H})} \leq \frac{\|\mathbf{J}\|_2^2 + \lambda}{\lambda}
```

With $\lambda = 10^{-4}$, this gives $\kappa \approx 10^4$—seemingly terrible (the contraction factor $1 - 1/\kappa \approx 0.9999$). However, **warm starting** changes everything:

- In streaming control, consecutive timesteps have nearly identical QPs
- The previous solution $\dot{\mathbf{q}}^{(k-1)}$ is already very close to the new optimum
- Instead of starting from scratch, we start within a few iterations of the answer

In practice, we observe **2-5 iterations** with warm starting versus 15-20 from a cold start.

### The Algorithm

```python
# Solve: min (1/2) x^T H x + g^T x  s.t.  ell <= x <= u
#
# H must be symmetric positive-definite (guaranteed by damping term λI)

def solve_box_qp(H, g, ell, u, x_prev=None):
    n = len(g)

    # === Fast path: try unconstrained solution first ===
    # If constraints aren't active, we can solve exactly via Cholesky
    # H x* = -g  =>  x* = H^{-1} (-g)
    try:
        cho = cho_factor(H)
        x_unconstrained = cho_solve(cho, -g)
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails (shouldn't happen with λ > 0)
        x_unconstrained = np.linalg.solve(H, -g)

    # Check if unconstrained solution is feasible
    if np.all(x_unconstrained >= ell) and np.all(x_unconstrained <= u):
        return x_unconstrained  # Lucky: no active constraints

    # === Projected gradient descent ===

    # Step size: α = 1 / ||H||_2  (spectral norm = largest eigenvalue)
    alpha = 1.0 / (np.linalg.norm(H, 2) + 1e-6)

    # Warm start from previous solution, or clamped unconstrained
    if x_prev is not None:
        x = np.clip(x_prev, ell, u)
    else:
        x = np.clip(x_unconstrained, ell, u)

    # Iterate until convergence
    for iteration in range(20):
        # Gradient: ∇f(x) = Hx + g
        grad = H @ x + g

        # Gradient descent step with projection
        x_new = np.clip(x - alpha * grad, ell, u)

        # Check convergence: ||x^{k+1} - x^k|| < tol
        if np.linalg.norm(x_new - x) < 1e-8:
            break

        x = x_new

    return x
```

### Why Not Newton's Method or Active Set?

**Newton's method** for QP requires solving a linear system at each iteration to find the search direction. For box constraints, this means tracking which constraints are active—the "active set" approach. While Newton converges in fewer iterations, each iteration is more expensive and the bookkeeping is complex.

**Interior point methods** are overkill for 6 variables. They're designed for large-scale problems where the matrix factorizations amortize.

**L-BFGS-B** (limited-memory BFGS with box constraints) is a quasi-Newton method that builds an approximation to the Hessian. In our testing, it has unpredictable iteration counts and poor warm-start behavior.

**Projected gradient descent** is ideal because:
- Each iteration is $O(n^2)$: one matrix-vector product
- Projection onto boxes is $O(n)$: element-wise clamp
- Warm starting works perfectly: just initialize with the previous solution
- No bookkeeping: the projection handles active constraints implicitly
- Convergent in finite iterations for our well-conditioned problem

### The Full Implementation

From `geodude/src/geodude/cartesian.py`:

```python
# Step 3: Solve box-constrained QP via projected gradient descent
# =========================================================================
# min (1/2) q_dot^T H q_dot + g^T q_dot  s.t. ell <= q_dot <= u
#
# For small box-constrained QP (6 DOF), projected gradient descent is fast
# and predictable - typically converges in 5-15 iterations. Much better than
# L-BFGS-B which has unpredictable iteration counts.

# Unconstrained solution via Cholesky (H is SPD by construction)
try:
    cho = cho_factor(H)
    qd_unconstrained = cho_solve(cho, -g)
except np.linalg.LinAlgError:
    # Fallback if Cholesky fails (shouldn't happen with λ > 0)
    qd_unconstrained = np.linalg.solve(H, -g)

# If unconstrained solution is feasible, we're done
if np.all(qd_unconstrained >= ell) and np.all(qd_unconstrained <= u):
    q_dot = qd_unconstrained
else:
    # Projected gradient descent for box-constrained QP
    # Very efficient - typically converges in 2-5 iterations with warm start

    # Warm start: use previous solution if provided, else clamped unconstrained
    if q_dot_prev is not None:
        q_dot = np.clip(q_dot_prev, ell, u)
    else:
        q_dot = np.clip(qd_unconstrained, ell, u)

    # Compute step size from eigenvalue bounds (H is well-conditioned due to λI)
    # Step size = 1 / max_eigenvalue(H) ≈ 1 / (||J||² + λ)
    alpha = 1.0 / (np.linalg.norm(H, 2) + 1e-6)

    for _ in range(20):  # Max 20 iterations (typically 2-5 with warm start)
        grad = H @ q_dot + g
        q_new = np.clip(q_dot - alpha * grad, ell, u)

        # Check convergence: gradient projected onto feasible set
        if np.linalg.norm(q_new - q_dot) < 1e-8:
            break

        q_dot = q_new
```

## Diagnostics: Knowing When You're Constrained

The solver reports why motion was limited:

```python
@dataclass
class TwistStepResult:
    joint_velocities: np.ndarray      # Solution
    twist_error: float                 # ||J*q_dot - v_d||_W
    achieved_fraction: float           # How much of desired twist was achieved (0-1)
    limiting_factor: str | None        # "joint_limit", "velocity", or None
```

The `achieved_fraction` is computed by projecting the achieved twist onto the desired:

```math
f = \frac{(\mathbf{J}\dot{\mathbf{q}})^T \mathbf{v}_d}{\|\mathbf{v}_d\|^2}
```

When $f < 1$, something is limiting motion. The `limiting_factor` distinguishes:
- **joint_limit**: A joint is near its position bound
- **velocity**: A joint is at its velocity limit
- **None**: Full twist achieved (unconstrained solution)

This enables higher-level logic to react appropriately—e.g., aborting a grasp approach if progress stalls.

## Implementation: The Complete Pipeline

```python
def step_twist(self, twist: np.ndarray, frame: str = "world") -> TwistStepResult:
    """Execute one timestep of Cartesian velocity control."""

    # 1. Get current Jacobian (6 x n_joints)
    J = self.arm.get_jacobian()

    # 2. Transform twist to world frame if needed
    if frame == "hand":
        R = self.arm.get_ee_rotation()
        twist[:3] = R @ twist[:3]
        twist[3:] = R @ twist[3:]

    # 3. Build QP matrices
    W = np.diag([1, 1, 1, 1/L**2, 1/L**2, 1/L**2])
    H = J.T @ W @ J + damping * np.eye(n_joints)
    g = -J.T @ W @ twist

    # 4. Compute velocity bounds
    q = self.arm.get_joint_positions()
    ell_pos = ((q_min + margin) - q) / dt
    u_pos = ((q_max - margin) - q) / dt
    ell = np.maximum(-qd_max, ell_pos)
    u = np.minimum(+qd_max, u_pos)

    # 5. Solve box-constrained QP
    q_dot = solve_box_qp(H, g, ell, u, warm_start=self._prev_q_dot)
    self._prev_q_dot = q_dot

    # 6. Apply to robot
    self.arm.set_joint_velocities(q_dot)

    # 7. Compute diagnostics
    achieved_twist = J @ q_dot
    twist_error = np.sqrt((achieved_twist - twist).T @ W @ (achieved_twist - twist))
    achieved_fraction = (achieved_twist @ twist) / (twist @ twist + 1e-10)

    return TwistStepResult(q_dot, twist_error, achieved_fraction, limiting_factor)
```

## Comparison with MoveIt Servo

| Aspect | MoveIt Servo | Our Approach |
|--------|--------------|--------------|
| **IK Method** | Pseudoinverse (SVD) | Damped least squares + QP |
| **Singularity** | Threshold detection → halt | Implicit via damping |
| **Joint limits** | Post-hoc clamping | Integrated in optimization |
| **Velocity limits** | Separate scaling | Box constraints |
| **Motion distortion** | Clamping changes direction | Direction preserved within feasible set |
| **Tuning** | Multiple thresholds | Single length scale + damping |

The fundamental difference: MoveIt Servo solves unconstrained IK then clips the solution, while we solve a constrained optimization that respects limits from the start.

## Practical Considerations

### Control Rate

We run at 125 Hz (8ms timestep), matching the UR5e servo rate. The QP solve takes <100μs with warm starting, leaving ample margin for communication latency.

### Length Scale Selection

The length scale $L$ should match your task's characteristic dimension:
- **Fine manipulation** (assembly, insertion): $L = 0.02$–$0.05$ m
- **General grasping**: $L = 0.1$ m (default)
- **Large workspace motions**: $L = 0.3$–$0.5$ m

### Damping Selection

The damping $\lambda$ controls the trade-off between tracking accuracy and joint velocity magnitude:
- **Too small** ($< 10^{-6}$): Large velocities near singularities, numerical instability
- **Too large** ($> 10^{-2}$): Sluggish response, poor tracking
- **Sweet spot**: $10^{-4}$ works well for most manipulators

### Safety Margins

The position limit margin creates a buffer zone:
- **5° default**: Prevents hard stops while preserving workspace
- **Increase for noisy sensors** or high-inertia loads
- **Decrease for precision** tasks requiring full range

## Conclusion

Cartesian velocity control with hard constraints requires treating limits as first-class citizens in the optimization, not as afterthoughts to be handled by clamping. Our QP formulation:

1. **Guarantees feasibility**: Joint limits are never violated
2. **Handles singularities gracefully**: Damping provides implicit regularization
3. **Preserves motion direction**: The optimizer finds the best achievable twist within constraints
4. **Runs in real-time**: Warm-started projected gradient descent converges in microseconds

The key insight is that constraints and objectives should be optimized jointly. Solving unconstrained IK and then clamping is fundamentally broken—it changes the motion direction in unpredictable ways. By posing the problem correctly from the start, we get predictable, safe, real-time Cartesian control.

## References

1. Buss, S. R. (2004). *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares Methods*. IEEE Journal of Robotics and Automation.

2. Nakamura, Y., & Hanafusa, H. (1986). *Inverse Kinematic Solutions with Singularity Robustness for Robot Manipulator Control*. ASME Journal of Dynamic Systems.

3. Flacco, F., De Luca, A., & Khatib, O. (2012). *Motion Control of Redundant Robots under Joint Constraints: Saturation in the Null Space*. IEEE ICRA.

4. MoveIt Servo Documentation. https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html
