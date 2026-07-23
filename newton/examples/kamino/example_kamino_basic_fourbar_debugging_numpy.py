# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for basic four-bar mechanism (NumPy / float64 reference solver)
#
# This is a debugging companion to ``example_kamino_basic_fourbar_debugging.py``.
# The model (a four-bar linkage spinning in zero gravity, with no joint limits
# and no contacts) is built exactly as in the reference example using the
# Kamino/Warp data structures, but the per-step forward-dynamics + integration
# loop is re-implemented from scratch in pure NumPy. The working precision is
# selectable via ``--precision`` (float64 by default, or float32).
#
# The intent is to isolate whether PADMM convergence issues observed in the
# float32 Warp solver are inherent to the algorithm or are float32 rounding
# artifacts: this file reproduces the exact same pipeline
# (SolverKaminoImpl._solve_forward_dynamics -> DualProblem -> PADMMSolver ->
# IntegratorEuler) in NumPy. Running it at float64 gives a double-precision
# reference; running it at float32 mirrors the Warp kernels' precision so the
# two effects can be told apart.
#
# Because the model has only bilateral joint constraints (no limits, no
# contacts), the PADMM feasible cone is all of R^n, so the projection is the
# identity, the De Saxce correction is zero, and the complementarity residual
# is zero. The full (non-accelerated, cold-started) PADMM iteration is still run
# so the convergence behavior can be compared against the float32 solver.
#
# Only the single-world case is supported.
#
# Command: python newton/examples/kamino/example_kamino_basic_fourbar_debugging_numpy.py --num-steps 100
#
###########################################################################

from __future__ import annotations

import argparse
import importlib.util
import math
import os

import numpy as np

import newton

# The Kamino builder used by the reference example lives in that example module.
# We load it dynamically to reuse ``build_boxes_fourbar`` verbatim rather than
# duplicating the ~340-line model construction.
_REF_PATH = os.path.join(os.path.dirname(__file__), "example_kamino_basic_fourbar_debugging.py")


def _load_build_boxes_fourbar():
    spec = importlib.util.spec_from_file_location("_kamino_fourbar_ref", _REF_PATH)
    module = importlib.util.module_from_spec(spec)
    # ``build_boxes_fourbar`` references ``math`` only on the ``limits=True`` path
    # (which we do not use), but inject it so the module imports cleanly regardless.
    module.math = math
    spec.loader.exec_module(module)
    return module.build_boxes_fourbar


###
# NumPy math helpers
#
# These mirror the Warp implementations in
# ``newton/_src/solvers/kamino/_src/core/math.py`` and
# ``.../kinematics/joints.py``. Quaternions follow the Warp convention:
# a 4-vector (x, y, z, w) with the scalar part last, unit norm, Hamilton product.
#
# The working precision is a module-level setting chosen via :func:`set_precision`,
# so every intermediate array below is constructed at the selected precision. At
# float32 this reproduces the Warp kernels' rounding; at float64 it provides a
# double-precision reference.
###


class _Precision:
    """Mutable holder for the current working NumPy precision and its epsilon."""

    def __init__(self, dtype: type = np.float64):
        self.dtype = np.dtype(dtype).type
        self.eps = np.finfo(self.dtype).eps


# Module-level working precision consulted by the math helpers. Defaults to
# float64; :func:`set_precision` switches it in place (no ``global`` needed).
_PREC = _Precision(np.float64)


def set_precision(dtype: type) -> None:
    """Select the working NumPy floating-point precision for the whole pipeline."""
    _PREC.dtype = np.dtype(dtype).type
    _PREC.eps = np.finfo(_PREC.dtype).eps


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric (cross-product) matrix of a 3-vector."""
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=_PREC.dtype,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a unit quaternion (x, y, z, w).

    Mirrors ``unit_quat_to_rotation_matrix`` in core/math.py.
    """
    x, y, z, w = q
    xx = 2.0 * x * x
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    wx = 2.0 * w * x
    yy = 2.0 * y * y
    yz = 2.0 * y * z
    wy = 2.0 * w * y
    zz = 2.0 * z * z
    wz = 2.0 * w * z
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=_PREC.dtype,
    )


def quat_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (x, y, z, w), matching core/math.py."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=_PREC.dtype,
    )


def quat_exp(v: np.ndarray) -> np.ndarray:
    """Exponential map of a rotation vector to a unit quaternion (x, y, z, w).

    Mirrors ``quat_exp`` in core/math.py.
    """
    vn = np.linalg.norm(v)
    if vn > _PREC.eps:
        a = 0.5 * vn
        sina = np.sin(a)
        cosa = np.cos(a)
        vu = v / vn
        return np.array([sina * vu[0], sina * vu[1], sina * vu[2], cosa], dtype=_PREC.dtype)
    return np.array([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0], dtype=_PREC.dtype)


def quat_box_plus(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q) [+] v == exp(v) * q, matching ``quat_box_plus`` in core/math.py."""
    return quat_product(quat_exp(v), q)


def screw_transform_matrix_from_points(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """6x6 screw transformation W_BA = [[I, 0], [skew(r_a - r_b), I]].

    Mirrors ``screw_transform_matrix_from_points`` in core/math.py.
    """
    w = np.eye(6, dtype=_PREC.dtype)
    w[3:6, 0:3] = skew(r_a - r_b)
    return w


def expand6d(x: np.ndarray) -> np.ndarray:
    """Block-diagonal 6x6 from a 3x3, matching ``expand6d`` in core/math.py."""
    out = np.zeros((6, 6), dtype=_PREC.dtype)
    out[0:3, 0:3] = x
    out[3:6, 3:6] = x
    return out


###
# The four-bar debugging model has a fixed joint layout (verified against the
# finalized ModelKamino): joint 0 is a FREE joint (world -> link1, 0 constraints)
# and joints 1..4 are REVOLUTE joints (5 kinematic constraints each). The
# revolute constraint axes select rows {T_x, T_y, T_z, R_y, R_z} of the local
# 6D screw, i.e. indices (0, 1, 2, 4, 5); see JointDoFType.REVOLUTE.cts_axes.
###
_REVOLUTE_CTS_AXES = (0, 1, 2, 4, 5)
_DOF_TYPE_FREE = 0
_DOF_TYPE_REVOLUTE = 1


class FourbarNumpySim:
    """Pure-NumPy reimplementation of the Kamino step for the four-bar.

    The working precision (``np.float64`` or ``np.float32``) is selected via the
    ``dtype`` argument and applied to the whole per-step pipeline.
    """

    def __init__(
        self,
        world_count: int = 1,
        overconstrained: bool = True,
        dtype: type = np.float64,
        delassus_regularization: float = 1e-3,
    ):
        if world_count != 1:
            raise ValueError("This NumPy debugging example only supports a single world.")

        # Select the working precision for all NumPy math helpers and store the
        # resolved scalar type / machine epsilon for use throughout the pipeline.
        set_precision(dtype)
        self.dtype = _PREC.dtype
        self.eps = _PREC.eps

        # CFM-style (constraint force mixing / Tikhonov compliance) diagonal
        # regularization added to the Delassus operator, i.e. we solve against
        # ``D + delta * I`` rather than the bare ``D``. The overconstrained
        # four-bar closes a kinematic loop with redundant revolute constraints,
        # so ``D = J M^-1 J^T`` is rank-deficient (here a 3D null space). Without
        # this term the PADMM iterate's null-space component is a pure accumulator
        # ``x_null += v_f_null / (eta + rho)`` that never reaches a fixed point,
        # so the dual residual floors above tolerance and the solve stalls at the
        # iteration cap. Unlike the proximal ``eta`` term (which also adds
        # ``eta * x_p`` to the right-hand side and thus cancels at the fixed
        # point), this term is NOT compensated on the RHS, so it makes the dual
        # objective strictly convex with a unique, bounded minimizer.
        #
        # The null-space multipliers satisfy ``J^T lambda = 0`` (they produce zero
        # net body wrench -- a self-cancelling internal preload), so a small delta
        # bounds the iterate with negligible effect on the trajectory. This mirrors
        # the real solver's ``DelassusOperator.set_regularization`` / ``_eta`` slot
        # (see dynamics/delassus.py). Set to 0.0 to reproduce the stalling behavior.
        self.delassus_regularization = float(delassus_regularization)

        # Time-stepping identical to the reference example.
        fps = 50
        frame_dt = 1.0 / fps
        substeps = max(1, round(frame_dt / 0.0025))
        self.dt = float(frame_dt / substeps)  # 0.0025 s

        # Build the model exactly as the reference example does, purely to obtain
        # a finalized ModelKamino, a consistent initial state, and the static
        # model data. After construction, the Warp solver is never stepped again.
        build_boxes_fourbar = _load_build_boxes_fourbar()

        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0
        build_boxes_fourbar(
            builder=robot_builder,
            floatingbase=True,
            dynamic_joints=False,
            implicit_pd=False,
            limits=False,
            overconstrained=overconstrained,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_world(robot_builder)
        model = builder.finalize(skip_validation_joints=True)
        model.gravity.fill_(0.0)

        # Initial velocities: rigid spin about the system center of mass, matching
        # the reference example.
        lin_vel_global = np.array([0.0, 0.0, 0.0])
        ang_vel_global = np.array([1.0, 1.0, 1.0]) * 10.0
        body_q_np = model.body_q.numpy()
        body_com_np = model.body_com.numpy()
        body_mass_np = model.body_mass.numpy()
        com = np.zeros((3,))
        mass = 0.0
        for body_id in range(model.body_count):
            com += body_mass_np[body_id] * (body_q_np[body_id, 0:3] + body_com_np[body_id, :])
            mass += body_mass_np[body_id]
        com = com / mass
        body_qd = []
        for body_id in range(model.body_count):
            lin_vel = lin_vel_global + np.cross(ang_vel_global, body_q_np[body_id, 0:3] - com)
            body_qd.append(np.concatenate((lin_vel, ang_vel_global)))
        model.body_qd.assign(body_qd)

        # Configure the Kamino solver identically to the reference example. The
        # config values (Baumgarte alpha, PADMM eta/rho/tolerances) are then read
        # back out and used by the NumPy loop.
        cfg = newton.solvers.SolverKamino.Config.from_model(model)
        cfg.use_collision_detector = False
        cfg.use_fk_solver = True
        cfg.collision_detector.pipeline = "primitive"
        cfg.collision_detector.max_contacts = 32 * model.world_count
        cfg.dynamics.preconditioning = True
        cfg.padmm.primal_tolerance = 1e-4
        cfg.padmm.dual_tolerance = 1e-4
        cfg.padmm.compl_tolerance = 1e-4
        cfg.padmm.max_iterations = 200
        cfg.padmm.rho_0 = 0.1
        cfg.padmm.use_acceleration = False
        cfg.padmm.warmstart_mode = "none"
        cfg.padmm.contact_warmstart_method = "geom_pair_net_force"
        solver = newton.solvers.SolverKamino(model=model, config=cfg)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Warm-start / reset sequence (identical to the reference example) to
        # produce a consistent initial state with joint velocities matching the
        # prescribed body velocities.
        solver.step(state_0, state_1, control, None, self.dt)
        solver.reset(state_0)
        state_0.body_qd.assign(model.body_qd)
        newton.eval_ik(model=model, state=state_0, joint_q=state_0.joint_q, joint_qd=state_0.joint_qd)

        # --- Extract static model data at the working precision -----------------
        mk = solver._solver_kamino._model

        assert mk.size.num_worlds == 1, "single-world only"
        self.num_bodies = int(mk.size.sum_of_num_bodies)
        self.num_joints = int(mk.size.sum_of_num_joints)
        self.num_cts = int(mk.info.num_joint_cts.numpy()[0])  # 20 bilateral constraints

        # Bodies
        self.inv_m = mk.bodies.inv_m_i.numpy().astype(self.dtype)  # (nb,)
        self.i_I = mk.bodies.i_I_i.numpy().astype(self.dtype)  # (nb, 3, 3) local inertia
        self.inv_i_I = mk.bodies.inv_i_I_i.numpy().astype(self.dtype)  # (nb, 3, 3)

        # Joints
        self.joint_dof_type = mk.joints.dof_type.numpy()
        self.joint_bid_B = mk.joints.bid_B.numpy()
        self.joint_bid_F = mk.joints.bid_F.numpy()
        self.joint_B_r_Bj = mk.joints.B_r_Bj.numpy().astype(self.dtype)
        self.joint_F_r_Fj = mk.joints.F_r_Fj.numpy().astype(self.dtype)
        self.joint_X_Bj = mk.joints.X_Bj.numpy().astype(self.dtype)  # (nj, 3, 3)
        self.joint_X_Fj = mk.joints.X_Fj.numpy().astype(self.dtype)
        self.joint_num_kin_cts = mk.joints.num_kinematic_cts.numpy()
        # Row offset of each joint's kinematic constraints within the total
        # constraints vector (== rows of the constraint Jacobian / Delassus).
        self.joint_kin_cts_row = mk.joints.kinematic_cts_offset_total_cts.numpy()

        # Solver / stabilization scalars (kept at Python float; cast where combined
        # with working-precision arrays).
        self.baumgarte_alpha = float(cfg.constraints.alpha)  # bilateral joint Baumgarte
        self.eta = float(cfg.padmm.eta)
        self.rho_0 = float(cfg.padmm.rho_0)
        self.primal_tol = float(cfg.padmm.primal_tolerance)
        self.dual_tol = float(cfg.padmm.dual_tolerance)
        self.max_iterations = int(cfg.padmm.max_iterations)
        self.preconditioning = bool(cfg.dynamics.preconditioning)

        # --- Snapshot the initial NumPy state -----------------------------------
        # body_q is stored [px, py, pz, qx, qy, qz, qw]. The bodies have zero CoM
        # offset (symmetric boxes), so the body-origin frame and CoM frame coincide.
        self.body_q = state_0.body_q.numpy().astype(self.dtype)  # (nb, 7)
        # body_qd / Kamino u_i uses the screw convention [linear (0:3), angular (3:6)].
        self.body_u = state_0.body_qd.numpy().astype(self.dtype)  # (nb, 6)

        self._step_id = 0

    ###
    # Per-step pipeline
    ###

    def _body_rotations(self) -> np.ndarray:
        """World-frame rotation matrix of each body from its pose quaternion."""
        return np.stack([quat_to_matrix(self.body_q[b, 3:7]) for b in range(self.num_bodies)])

    def _world_inertias(self, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-frame inertia I_i = R I_local R^T and its inverse per body.

        Mirrors ``transform_body_inertial_properties`` in core/bodies.py.
        """
        I_w = np.empty((self.num_bodies, 3, 3), dtype=self.dtype)
        inv_I_w = np.empty((self.num_bodies, 3, 3), dtype=self.dtype)
        for b in range(self.num_bodies):
            R = rot[b]
            I_w[b] = R @ self.i_I[b] @ R.T
            inv_I_w[b] = R @ self.inv_i_I[b] @ R.T
        return I_w, inv_I_w

    def _joint_kinematics(self, rot: np.ndarray):
        """Compute, per joint, the joint-frame world position and orientation, and
        the kinematic constraint residual.

        Mirrors ``compute_joint_pose_and_relative_motion`` (kinematics/joints.py)
        combined with the revolute residual write in ``make_typed_write_joint_data``.

        Returns a list of per-joint dicts (only for constraint-bearing joints).
        """
        joints = []
        for j in range(self.num_joints):
            if self.joint_dof_type[j] != _DOF_TYPE_REVOLUTE:
                # The FREE base joint contributes no constraints and no Jacobian rows.
                continue

            bid_B = int(self.joint_bid_B[j])
            bid_F = int(self.joint_bid_F[j])
            X_Bj = self.joint_X_Bj[j]
            X_Fj = self.joint_X_Fj[j]

            # Base body pose (identity/world if bid_B < 0). For this model every
            # revolute joint is binary, but keep the world case for generality.
            if bid_B > -1:
                r_B = self.body_q[bid_B, 0:3]
                R_B = rot[bid_B]
            else:
                r_B = np.zeros(3, dtype=self.dtype)
                R_B = np.eye(3, dtype=self.dtype)
            r_F = self.body_q[bid_F, 0:3]
            R_F = rot[bid_F]

            # Joint frame orientation on either body, in world coordinates.
            R_Bj = R_B @ X_Bj  # == R(q_Bj)
            R_Fj = R_F @ X_Fj  # == R(q_Fj)

            # Joint-frame anchor positions in world coordinates.
            r_Bj = R_B @ self.joint_B_r_Bj[j]
            r_Fj = R_F @ self.joint_F_r_Fj[j]
            r_j_B = r_B + r_Bj  # p_j translation (used by the Jacobian screw transforms)
            r_j_F = r_F + r_Fj
            gap = r_j_F - r_j_B  # world-frame separation of the two anchor frames

            # Relative rotation, base joint frame -> follower joint frame.
            R_rel = R_Bj.T @ R_Fj  # == R(j_q_j)

            # Translational residual expressed in the base joint frame.
            j_r_j = R_Bj.T @ gap

            # Angular residual for a revolute joint: cross(x_B, x_F) with x_B = e_x,
            # x_F = R_rel @ e_x (== first column of R_rel).
            x_F = R_rel[:, 0]
            j_theta = np.cross(np.array([1.0, 0.0, 0.0], dtype=self.dtype), x_F)

            # 6D local residual screw = [j_r_j (linear), j_theta (angular)], then
            # select the revolute constraint axes (0, 1, 2, 4, 5).
            res6 = np.array([j_r_j[0], j_r_j[1], j_r_j[2], j_theta[0], j_theta[1], j_theta[2]], dtype=self.dtype)
            residual = res6[list(_REVOLUTE_CTS_AXES)]

            joints.append(
                {
                    "jid": j,
                    "bid_B": bid_B,
                    "bid_F": bid_F,
                    "R_Bj": R_Bj,  # joint-frame world orientation (base side)
                    "r_j": r_j_B,  # joint-frame world position
                    "r_B": r_B,
                    "r_F": r_F,
                    "row0": int(self.joint_kin_cts_row[j]),
                    "residual": residual,
                }
            )
        return joints

    def _build_constraint_jacobian(self, joints) -> np.ndarray:
        """Assemble the dense constraint Jacobian J_cts of shape (num_cts, 6*nb).

        Mirrors ``_build_joint_jacobians_dense`` (kinematics/jacobians.py): for each
        joint the extended per-body transposed Jacobians are
        ``JT_B = -W_j_B @ expand6d(R_Bj)`` and ``JT_F = W_j_F @ expand6d(R_Bj)``,
        and constraint row ``a`` receives column ``a`` of these blocks in the
        follower/base body 6-column slots.
        """
        nbd = 6 * self.num_bodies
        J = np.zeros((self.num_cts, nbd), dtype=self.dtype)
        for jd in joints:
            R_bar = expand6d(jd["R_Bj"])
            W_j_B = screw_transform_matrix_from_points(jd["r_j"], jd["r_B"])
            W_j_F = screw_transform_matrix_from_points(jd["r_j"], jd["r_F"])
            JT_B = -W_j_B @ R_bar  # (6, 6)
            JT_F = W_j_F @ R_bar  # (6, 6)
            bid_B = jd["bid_B"]
            bid_F = jd["bid_F"]
            row0 = jd["row0"]
            for k, axis in enumerate(_REVOLUTE_CTS_AXES):
                row = row0 + k
                J[row, 6 * bid_F : 6 * bid_F + 6] = JT_F[:, axis]
                if bid_B > -1:
                    J[row, 6 * bid_B : 6 * bid_B + 6] = JT_B[:, axis]
        return J

    def _apply_inv_mass(self, u: np.ndarray, inv_I_w: np.ndarray) -> np.ndarray:
        """Apply the block-diagonal inverse generalized mass matrix M^{-1} to a
        stacked body-space vector (shape (6*nb,)).

        M^{-1} per body = diag(inv_m * I3, inv_I_world).
        """
        out = np.empty_like(u)
        for b in range(self.num_bodies):
            f = u[6 * b : 6 * b + 3]
            tau = u[6 * b + 3 : 6 * b + 6]
            out[6 * b : 6 * b + 3] = self.inv_m[b] * f
            out[6 * b + 3 : 6 * b + 6] = inv_I_w[b] @ tau
        return out

    def _free_generalized_velocity(self, I_w: np.ndarray, inv_I_w: np.ndarray) -> np.ndarray:
        """Unconstrained next generalized velocity u_f (shape (6*nb,)).

        Mirrors ``_build_generalized_free_velocity`` (dynamics/dual.py) with zero
        gravity, external, and actuation wrenches: only the gyroscopic (Coriolis)
        term contributes. ``u_f = u + dt * M^{-1} @ h`` with
        ``h = [0, -skew(omega) @ (I omega)]``.
        """
        u_f = np.empty(6 * self.num_bodies, dtype=self.dtype)
        for b in range(self.num_bodies):
            v = self.body_u[b, 0:3]
            omega = self.body_u[b, 3:6]
            tau_cor = -skew(omega) @ (I_w[b] @ omega)
            u_f[6 * b : 6 * b + 3] = v  # + dt * inv_m * f, with f = 0
            u_f[6 * b + 3 : 6 * b + 6] = omega + self.dt * (inv_I_w[b] @ tau_cor)
        return u_f

    def _padmm_solve(self, D: np.ndarray, v_f: np.ndarray, P: np.ndarray):
        """Non-accelerated, cold-started PADMM iteration on the bilateral system.
        Also takes the preconditioning as input.
        bilateral system.

        Mirrors ``PADMMSolver.solve`` / ``_step`` for the pure-bilateral case
        (no limits, no contacts): the projection onto the feasible cone is the
        identity, the De Saxce correction ``s`` is zero, and the complementarity
        residual is zero. The regularized system ``D_reg = D_hat + (eta + rho) I``
        is factorized once (fixed penalty).

        Returns (y, z, iterations, r_p_inf, r_d_inf).
        """
        n = self.num_cts
        eta = self.eta
        rho = self.rho_0  # fixed penalty (penalty_update_method == "fixed")

        # CFM-style diagonal regularization of the Delassus operator (see the
        # ``delassus_regularization`` note in ``__init__``). This is folded into the
        # (already diagonal) proximal + ALM term of ``D_reg`` since the whole system
        # is preconditioned uniformly; it is deliberately NOT added to the RHS ``v``
        # below, which is what makes the dual objective strictly convex and bounds
        # the otherwise-accumulating null-space component of the iterate.
        delta = self.delassus_regularization

        # D_reg := D_hat + (eta + rho + delta) I  (proximal + ALM + CFM regularization).
        D_hat = (P[:, None] * D) * P[None, :]
        D_reg = D_hat + (eta + rho + delta) * np.eye(n, dtype=self.dtype)
        L = np.linalg.cholesky(D_reg)

        v_f_hat = P * v_f

        eigvals, eigvecs = np.linalg.eigh(D_hat)
        tol = 1e-8
        range_basis = eigvecs[:, eigvals > tol]
        null_basis = eigvecs[:, eigvals <= tol]

        # Project v_f_hat onto range(D_hat)
        # v_f_hat -= null_basis @ (null_basis.T @ v_f_hat)

        # Cold start: all state variables zero.
        x_p = np.zeros(n, dtype=self.dtype)
        y_p = np.zeros(n, dtype=self.dtype)
        z_p = np.zeros(n, dtype=self.dtype)

        iterations = 0
        r_p_inf = np.inf
        r_d_inf = np.inf
        for i in range(self.max_iterations):  # noqa: B007  (i is used by the commented-out debug loads below)
            # Velocity bias: v = -v_f - s + eta*x_p + rho*y_p + z_p, with s = 0.
            v = -v_f_hat + eta * x_p + rho * y_p + z_p

            # Unconstrained sub-problem: x = D_reg^{-1} v (Cholesky solve).
            x = _chol_solve(L, v)

            print("Range basis residual:", np.max(np.abs(range_basis.T @ (x - x_p))))
            print("Null basis residual:", np.max(np.abs(null_basis.T @ (x - x_p))))

            # print("Original problem residual:", np.max(np.abs(D_hat @ x + v_f_hat)))

            # Projection argument then projection onto the feasible cone.
            # With only bilateral constraints the cone is R^n, so projection is
            # the identity: y = x - (1/rho) * z_p.
            y = x - (1.0 / rho) * z_p

            # Dual update and primal/dual residuals.
            z = z_p + rho * (y - x)
            # r_p = y - x  # preconditioner P applied below when reporting norms
            # r_d = eta * (x - x_p) + rho * (y - y_p)
            r_p = P * (x - y)  # preconditioner P applied below when reporting norms
            r_d = (1 / P) * (eta * (x - x_p) + rho * (y - y_p))

            # print("x - x_p:", x - x_p)

            iterations += 1
            r_p_inf = float(np.max(np.abs(r_p))) if n > 0 else 0.0
            r_d_inf = float(np.max(np.abs(r_d))) if n > 0 else 0.0

            # print("Dual residual:", r_d_inf)

            # Convergence check (r_c == 0 for bilateral-only problems). Matches the
            # ``iterations > 1`` guard in the Warp convergence kernel.
            if iterations > 1 and r_p_inf <= self.primal_tol and r_d_inf <= self.dual_tol:
                x_p, y_p, z_p = x, y, z
                break

            x_p, y_p, z_p = x, y, z

        return P * y_p, (1 / P) * z_p, iterations, r_p_inf, r_d_inf

    def step(self):
        """Advance the state by one time step dt at the configured precision."""
        # Re-assert the working precision in case another sim instance changed the
        # module-level setting since this one was constructed.
        set_precision(self.dtype)

        rot = self._body_rotations()
        I_w, inv_I_w = self._world_inertias(rot)

        # 1) Joint kinematics: residuals + joint-frame poses.
        joints = self._joint_kinematics(rot)

        # 2) Constraint Jacobian.
        J = self._build_constraint_jacobian(joints)

        # 3) Free (unconstrained) generalized velocity.
        u_f = self._free_generalized_velocity(I_w, inv_I_w)

        # 4) Velocity bias v_b = alpha * inv_dt * r_j (kinematic joint rows only).
        inv_dt = 1.0 / self.dt
        v_b = np.zeros(self.num_cts, dtype=self.dtype)
        for jd in joints:
            row0 = jd["row0"]
            n_kin = int(self.joint_num_kin_cts[jd["jid"]])
            v_b[row0 : row0 + n_kin] = self.baumgarte_alpha * inv_dt * jd["residual"]

        # 5) Delassus matrix D = J M^{-1} J^T and free constraint velocity v_f.
        Minv_JT = np.empty_like(J.T)
        for r in range(self.num_cts):
            Minv_JT[:, r] = self._apply_inv_mass(J[r, :], inv_I_w)
        D = J @ Minv_JT
        D = 0.5 * (D + D.T)  # symmetrize (mirrors the upper-triangle mirror in Warp)
        v_f = J @ u_f + v_b

        # 6) Diagonal Jacobi preconditioner P_i = sqrt(1 / (|D_ii| + eps)).
        if self.preconditioning:
            P = np.sqrt(1.0 / (np.abs(np.diag(D)) + self.eps))
        else:
            P = np.ones(self.num_cts, dtype=self.dtype)
        # D_hat = (P[:, None] * D) * P[None, :]
        # v_f_hat = P * v_f

        # 7) PADMM solve (preconditioned space).
        y_hat, _z_hat, iters, r_p_inf, r_d_inf = self._padmm_solve(D, v_f, P)

        # 8) Undo preconditioning: lambda = P * y, v_plus = z / P.
        # lambdas = P * y_hat  # constraint impulses
        lambdas = y_hat  # constraint impulses

        # 9) Constraint body wrenches: w = J^T @ (inv_dt * lambda), per body.
        w_total = J.T @ (inv_dt * lambdas)  # (6*nb,), total wrench (only w_j is nonzero)

        # 10) Semi-implicit Euler integration (IntegratorEuler / core.math EOM).
        for b in range(self.num_bodies):
            f = w_total[6 * b : 6 * b + 3]
            tau = w_total[6 * b + 3 : 6 * b + 6]
            v = self.body_u[b, 0:3]
            omega = self.body_u[b, 3:6]

            # Twist update with the maximal-coordinate equations of motion
            # (gravity g = 0). The Coriolis term is added here, matching
            # ``compute_body_twist_update_with_eom``.
            v_n = v + self.dt * (self.inv_m[b] * f)
            tau_gyro = -skew(omega) @ (I_w[b] @ omega)
            omega_n = omega + self.dt * (inv_I_w[b] @ (tau_gyro + tau))

            # Pose update with the exponential map (``compute_body_pose_update_with_logmap``).
            r = self.body_q[b, 0:3]
            q = self.body_q[b, 3:7]
            r_n = r + self.dt * v_n
            q_n = quat_box_plus(q, self.dt * omega_n)

            self.body_q[b, 0:3] = r_n
            self.body_q[b, 3:7] = q_n
            self.body_u[b, 0:3] = v_n
            self.body_u[b, 3:6] = omega_n

        self._step_id += 1

        # Return per-step diagnostics.
        max_residual = max((float(np.max(np.abs(jd["residual"]))) for jd in joints), default=0.0)
        return {
            "iterations": iters,
            "r_primal": r_p_inf,
            "r_dual": r_d_inf,
            "max_constraint_residual": max_residual,
        }

    ###
    # Diagnostics
    ###

    def total_angular_momentum(self) -> np.ndarray:
        """Total angular momentum about the origin (world frame), for a sanity
        check: with no external wrench it should be (approximately) conserved."""
        set_precision(self.dtype)
        rot = self._body_rotations()
        I_w, _ = self._world_inertias(rot)
        L = np.zeros(3, dtype=self.dtype)
        m = 1.0 / self.inv_m
        for b in range(self.num_bodies):
            r = self.body_q[b, 0:3]
            v = self.body_u[b, 0:3]
            omega = self.body_u[b, 3:6]
            L += I_w[b] @ omega + m[b] * np.cross(r, v)
        return L

    def max_speed(self) -> float:
        return float(np.max(np.abs(self.body_u)))


def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L L^T) x = b given a lower Cholesky factor L."""
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-steps", type=int, default=5, help="Number of simulation steps to run.")
    parser.add_argument("--world-count", type=int, default=1, help="Number of worlds (must be 1).")
    parser.add_argument(
        "--overconstrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the overconstrained (all-revolute) version of the four-bar model.",
    )
    parser.add_argument(
        "--precision",
        choices=["float64", "float32"],
        default="float64",
        help="Working NumPy precision for the simulation loop. Defaults to float64.",
    )
    parser.add_argument(
        "--delassus-regularization",
        type=float,
        default=0.0,
        help=(
            "CFM-style diagonal regularization added to the Delassus operator to make the "
            "redundant-constraint (rank-deficient) system converge. Set to 0 to reproduce the "
            "stalling behavior."
        ),
    )
    parser.add_argument("--print-every", type=int, default=1, help="Print diagnostics every N steps.")
    args = parser.parse_args()

    dtype = np.float64 if args.precision == "float64" else np.float32
    sim = FourbarNumpySim(
        world_count=args.world_count,
        overconstrained=args.overconstrained,
        dtype=dtype,
        delassus_regularization=args.delassus_regularization,
    )

    print(
        f"NumPy {args.precision} Kamino four-bar: dt={sim.dt:g}, num_cts={sim.num_cts}, max_iter={sim.max_iterations}"
    )
    L0 = sim.total_angular_momentum()
    print(f"initial |angular momentum| = {np.linalg.norm(L0):.6e}")

    for i in range(args.num_steps):
        info = sim.step()
        if args.print_every > 0 and (i % args.print_every == 0 or i == args.num_steps - 1):
            L = sim.total_angular_momentum()
            print(
                f"step {i:4d} | padmm iters {info['iterations']:4d} | "
                f"r_p {info['r_primal']:.3e} r_d {info['r_dual']:.3e} | "
                f"max cts residual {info['max_constraint_residual']:.3e} | "
                f"max speed {sim.max_speed():.4f} | |L| {np.linalg.norm(L):.6e}"
            )

    Lf = sim.total_angular_momentum()
    print(f"final |angular momentum| = {np.linalg.norm(Lf):.6e} (drift {np.linalg.norm(Lf - L0):.3e})")
    if not np.all(np.isfinite(sim.body_q)) or not np.all(np.isfinite(sim.body_u)):
        raise RuntimeError("Simulation diverged: non-finite state detected.")


if __name__ == "__main__":
    main()
