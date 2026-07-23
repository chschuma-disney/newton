# Four-bar PADMM convergence: rank-deficiency and the CFM fix

This note documents a convergence failure observed in
`example_kamino_basic_fourbar_debugging_numpy.py` (the float64 NumPy
reimplementation of the Kamino PADMM forward-dynamics step), why the obvious
remedies do not work, and what the final fix does. The same analysis applies to
the float32 Warp/Kamino solver — the NumPy port reproduces its behavior exactly.

## Symptom

Running the overconstrained four-bar (zero gravity, no limits, no contacts), the
PADMM solve converges for the first couple of steps and then stalls, hitting the
iteration cap (`max_iterations = 200`) from step 2 onward. The dual residual does
not decrease monotonically within a step: it drops for the first few iterations,
then plateaus flat just above the `1e-4` tolerance for the remaining ~195
iterations.

```
step 0 | iters   6 | r_d 1.13e-07   (converged)
step 1 | iters   9 | r_d 9.89e-07   (converged)
step 2 | iters 200 | r_d 1.23e-05   (stalled)
step 3 | iters 200 | r_d 4.61e-05   (stalled)
step 4 | iters 200 | r_d 1.14e-04   (stalled)
step 5 | iters 200 | r_d 2.29e-04   (stalled)
```

## Root cause: a rank-deficient Delassus operator

The overconstrained four-bar closes a kinematic loop with four revolute joints,
contributing `4 x 5 = 20` bilateral constraints. But the loop is
**overconstrained**: those constraints are not independent. The constraint
Jacobian has rank only 17-18 of 20, so the preconditioned Delassus operator

```
D_hat = P (J M^-1 J^T) P
```

has a **3-dimensional null space**. Measured spectrum: three eigenvalues at
~1e-16 and a maximum of ~2.0.

The solver never operates on the bare `D_hat`. It solves the proximal + ALM
regularized system

```
D_reg = D_hat + (eta + rho) I
```

whose condition number is fine (~21), so the linear solve itself is well posed.
The problem is not the linear solve — it is the ADMM fixed-point iteration built
on top of it.

## Why the iteration stalls: null-space accumulation

For a purely bilateral problem (no limits, no contacts) the feasible cone is all
of R^n, so the projection is the identity. That collapses the PADMM step to:

```
y = x - z_p / rho
z = z_p + rho (y - x)      ==>   z == 0  and  y == x   for all iterations after the first
x_{k+1} = D_reg^-1 ( -v_f_hat + (eta + rho) x_k )
```

Split `x` into the range space and null space of `D_hat`. On the **range space**
the map is a contraction and `x_range` converges normally (settles within ~5
iterations). On the **null space**, `D_hat` acts as zero, so `D_reg` acts as
`(eta + rho) I`, and the update degenerates to a pure accumulator:

```
x_null,{k+1} = x_null,k + v_f_hat_null / (eta + rho)
```

If the right-hand side has **any** component `v_f_hat_null` in the null space,
`x_null` ramps by a constant amount every iteration and never reaches a fixed
*point* — only a fixed *step*. The dual residual depends on `x - x_p`, which
therefore floors at a nonzero constant instead of decaying to zero.

Measured inside a stalled step: `||x_range||` stabilizes at ~9.49e-02 by
iteration 5, while `||x_null||` keeps growing 1.2e-5 -> 2.5e-5 -> 3.7e-5 -> ...
linearly and without bound.

### Why it converges early and stalls later

The null-space RHS component `v_f_hat_null` grows as loop-closure drift
accumulates over the simulation:

| step | `||v_f_hat_null||` | outcome    |
|------|--------------------|------------|
| 0    | 4e-18 (~machine 0) | converges  |
| 1    | 1e-8               | converges  |
| 2    | 1e-7               | stalls     |
| 3    | 5e-7               | stalls     |
| 4    | 1e-6               | stalls     |

While `v_f_hat_null` is at machine-zero the ramp is invisible; once it crosses
~1e-7 the per-iteration null step exceeds the tolerance and the solve stalls.

### The stall is physically benign

The null-space multipliers `lambda` satisfy

```
lambda^T D lambda = || M^{-1/2} J^T lambda ||^2 = 0   ==>   J^T lambda = 0
```

i.e. they produce **zero net body wrench** — a self-cancelling internal preload
in the redundant constraints. That is why the simulated trajectory already looks
correct despite the residual flag: only the arbitrary internal-preload part of
the solution drifts, not the physically meaningful part.

## Why the obvious remedies do not work

**Increasing `eta` or `rho`.** In the stalled regime the dual residual on the
null space is

```
r_d_null = (1/P) (eta + rho) (x_null - x_p,null)
         = (1/P) (eta + rho) [ v_f_hat_null / (eta + rho) ]
         = (1/P) v_f_hat_null
```

The penalty `(eta + rho)` **cancels**. The residual floor equals the null-space
RHS magnitude and is completely independent of `eta` and `rho`. Larger values
shrink the per-iteration ramp step but leave the floor unchanged — no
convergence.

**Adaptive / balanced penalty.** Here `r_p == 0` identically, so a balancer that
compares primal vs dual residuals keeps driving `rho -> rho_min`, worsening
conditioning. Actively harmful.

**Nesterov acceleration.** The dual objective is unbounded below along the null
direction — there is no fixed point to accelerate toward. Acceleration only
speeds convergence to a minimizer that does not exist, and tends to amplify the
ramp.

The only quantity that moves the residual floor is `v_f_hat_null` itself, so the
fix must change the operator, not the penalty.

## The fix: CFM-style Delassus regularization

Solve against a genuinely regularized operator

```
D_reg = D_hat + (eta + rho + delta) I
```

with a small `delta > 0` (default `1e-2`), **without** adding any compensating
term to the right-hand side.

This is the crucial distinction from the proximal `eta` term. The proximal term
adds `(eta + rho) I` to the matrix *and* `eta x_p` to the RHS; those cancel at
the fixed point, which is exactly why it cannot bound the null-space component. A
CFM (constraint-force-mixing / Tikhonov compliance) term adds `delta I` to the
operator **only**. That makes the dual objective

```
1/2 x^T (D_hat + delta I) x + x^T v_f_hat
```

strictly convex with a unique, bounded minimizer `x_null* = -v_f_hat_null /
delta`. ADMM now converges to a true fixed point on both subspaces.

On the null space the update becomes a contraction with rate
`(eta + rho) / (eta + rho + delta)`, so `delta` must be an appreciable fraction
of `(eta + rho)` (here `0.1`) to converge within the iteration budget. The cost
is a small constraint compliance of order `delta` and a correspondingly small,
bounded internal preload.

### Why this is the right general fix

- **General:** works for any rank-deficiency pattern without needing to know the
  null space explicitly (unlike projecting the RHS onto `range(D)`, which needs a
  per-step SVD/eig and is incompatible with the matrix-free sparse operator).
- **Cheap and matrix-free:** a diagonal add, exactly the form the sparse Delassus
  operator already supports via `DelassusOperator.set_regularization` / the
  `_eta` slot (`dynamics/delassus.py`), so it ports to the Warp solver at O(n).
- **Standard practice:** this is CFM in ODE and the regularization term in
  MuJoCo/PhysX. Overconstrained linkages are *meant* to be resolved by a small
  compliance, yielding a well-defined minimum-energy preload distribution instead
  of an ill-posed one.
- **Physically negligible:** because the null-space multipliers carry zero net
  wrench, `delta` can be small with negligible effect on the trajectory.

### Verified effect

| | `delta = 1e-2` (fix) | `delta = 0` (baseline) |
|-----------------------|----------------------|------------------------|
| step 2 iterations     | 29                   | 200 (cap)              |
| step 4 iterations     | 51                   | 200 (cap)              |
| step 7 iterations     | 69                   | 200 (cap)              |
| step 4 dual residual  | 9.98e-07 (converged) | 1.14e-04 (stalled)     |
| step 4 constraint res | 2.136e-04            | 2.077e-04              |
| trajectory change     | ~3.4e-4 rel. velocity (0.03%) | -             |

Every step converges well under the `1e-4` tolerance, the physical constraint
residual and angular momentum are unchanged, and the trajectory shift is
negligible.

## Reproducing and tuning

```bash
# Fix on (default)
python newton/examples/kamino/example_kamino_basic_fourbar_debugging_numpy.py --num-steps 8

# Reproduce the stall
python newton/examples/kamino/example_kamino_basic_fourbar_debugging_numpy.py --num-steps 8 --delassus-regularization 0

# Sweep the compliance
python newton/examples/kamino/example_kamino_basic_fourbar_debugging_numpy.py --num-steps 8 --delassus-regularization 1e-3
```

`delta` trades convergence speed against constraint stiffness: too small and the
null space is not lifted above numerical noise (no convergence in budget); too
large and constraints become noticeably soft. A value that is a small fraction of
`(eta + rho)`, or scaled by `trace(D)/n`, covers the general case.
