# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for basic four-bar mechanism
#
# Shows how to simulate a basic four-bar linkage with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_basic_fourbar --world-count 16
#
###########################################################################

import argparse
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver
from newton._src.solvers.kamino.tests.utils.extract import extract_info_vectors


def build_boxes_fourbar(
    builder: newton.ModelBuilder | None = None,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    floatingbase: bool = False,
    limits: bool = True,
    dynamic_joints: bool = False,
    implicit_pd: bool = False,
    verbose: bool = False,
    actuator_ids: list[int] | None = None,
    overconstrained: bool = True,
    fixed: bool = False,
) -> newton.ModelBuilder:
    from newton._src.geometry import inertia  # noqa: PLC0415

    # Create a new builder if none is provided
    if builder is None:
        _builder = newton.ModelBuilder()
    else:
        _builder = builder

    _builder.begin_world(label="boxes_fourbar")

    # Set default actuator IDs if none are provided
    if actuator_ids is None:
        actuator_ids = [1, 3]
    elif not isinstance(actuator_ids, list):
        raise TypeError("actuator_ids, if specified, must be provided as a list of integers.")

    ###
    # Base Parameters
    ###

    # Constant to set an initial z offset for the bodies
    # NOTE: for testing purposes, recommend values are {0.0, -0.001}
    z_0 = z_offset

    # Box dimensions
    d = 0.01
    w = 0.01
    h = 0.1

    # Margins
    mj = 0.001
    dj = 0.5 * d + mj

    ###
    # Body parameters
    ###

    # Box dimensions
    d_1 = h
    w_1 = w
    h_1 = d
    d_2 = d
    w_2 = w
    h_2 = h
    d_3 = h
    w_3 = w
    h_3 = d
    d_4 = d
    w_4 = w
    h_4 = h

    # Inertial properties
    m_i = 1.0
    i_I_i_1 = inertia.compute_inertia_box_from_mass(mass=m_i, hx=0.5 * d_1, hy=0.5 * w_1, hz=0.5 * h_1)
    i_I_i_2 = inertia.compute_inertia_box_from_mass(mass=m_i, hx=0.5 * d_2, hy=0.5 * w_2, hz=0.5 * h_2)
    i_I_i_3 = inertia.compute_inertia_box_from_mass(mass=m_i, hx=0.5 * d_3, hy=0.5 * w_3, hz=0.5 * h_3)
    i_I_i_4 = inertia.compute_inertia_box_from_mass(mass=m_i, hx=0.5 * d_4, hy=0.5 * w_4, hz=0.5 * h_4)
    if verbose:
        print(f"i_I_i_1:\n{i_I_i_1}")
        print(f"i_I_i_2:\n{i_I_i_2}")
        print(f"i_I_i_3:\n{i_I_i_3}")
        print(f"i_I_i_4:\n{i_I_i_4}")

    # Initial body positions
    r_0 = wp.vec3f(0.0, 0.0, z_0)
    dr_b1 = wp.vec3f(0.0, 0.0, 0.5 * d)
    dr_b2 = wp.vec3f(0.5 * h + dj, 0.0, 0.5 * h + dj)
    dr_b3 = wp.vec3f(0.0, 0.0, 0.5 * d + h + dj + mj)
    dr_b4 = wp.vec3f(-0.5 * h - dj, 0.0, 0.5 * h + dj)

    # Initial positions of the bodies
    r_b1 = r_0 + dr_b1
    r_b2 = r_b1 + dr_b2
    r_b3 = r_b1 + dr_b3
    r_b4 = r_b1 + dr_b4
    if verbose:
        print(f"r_b1: {r_b1}")
        print(f"r_b2: {r_b2}")
        print(f"r_b3: {r_b3}")
        print(f"r_b4: {r_b4}")

    # Initial body poses
    q_i_1 = wp.transformf(r_b1, wp.quat_identity(dtype=wp.float32))
    q_i_2 = wp.transformf(r_b2, wp.quat_identity(dtype=wp.float32))
    q_i_3 = wp.transformf(r_b3, wp.quat_identity(dtype=wp.float32))
    q_i_4 = wp.transformf(r_b4, wp.quat_identity(dtype=wp.float32))

    # Initial joint positions
    r_j1 = wp.vec3f(r_b2.x, 0.0, r_b1.z)
    r_j2 = wp.vec3f(r_b2.x, 0.0, r_b3.z)
    r_j3 = wp.vec3f(r_b4.x, 0.0, r_b3.z)
    r_j4 = wp.vec3f(r_b4.x, 0.0, r_b1.z)
    if verbose:
        print(f"r_j1: {r_j1}")
        print(f"r_j2: {r_j2}")
        print(f"r_j3: {r_j3}")
        print(f"r_j4: {r_j4}")

    ###
    # Bodies
    ###

    bid1 = _builder.add_link(
        label="link_1",
        mass=m_i,
        inertia=i_I_i_1,
        xform=q_i_1,
        lock_inertia=True,
    )

    bid2 = _builder.add_link(
        label="link_2",
        mass=m_i,
        inertia=i_I_i_2,
        xform=q_i_2,
        lock_inertia=True,
    )

    bid3 = _builder.add_link(
        label="link_3",
        mass=m_i,
        inertia=i_I_i_3,
        xform=q_i_3,
        lock_inertia=True,
    )

    bid4 = _builder.add_link(
        label="link_4",
        mass=m_i,
        inertia=i_I_i_4,
        xform=q_i_4,
        lock_inertia=True,
    )

    ###
    # Geometries
    ###

    # Add collision geometries
    _builder.add_shape_box(
        label="box_1",
        body=bid1,
        hx=0.5 * d_1,
        hy=0.5 * w_1,
        hz=0.5 * h_1,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )
    _builder.add_shape_box(
        label="box_2",
        body=bid2,
        hx=0.5 * d_2,
        hy=0.5 * w_2,
        hz=0.5 * h_2,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )
    _builder.add_shape_box(
        label="box_3",
        body=bid3,
        hx=0.5 * d_3,
        hy=0.5 * w_3,
        hz=0.5 * h_3,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )
    _builder.add_shape_box(
        label="box_4",
        body=bid4,
        hx=0.5 * d_4,
        hy=0.5 * w_4,
        hz=0.5 * h_4,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    ###
    # Joints
    ###

    # Revolute joint position limits
    if limits:
        qmin = -0.25 * math.pi
        qmax = 0.25 * math.pi
    else:
        qmin = float(-newton.MAXVAL)
        qmax = float(newton.MAXVAL)

    # List of articulation joints
    articulation_joints = []

    # Optional fixed base: attach link_1 rigidly to the world
    if fixedbase:
        _builder.add_joint_fixed(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transformf(-r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    # Optional floating base: attach link_1 to the world with a 6-DoF free joint
    if floatingbase:
        jf = _builder.add_joint_free(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )
        articulation_joints.append(jf)

    # Per-DoF configurations reused across the revolute joints
    passive_joint_dof_config = newton.ModelBuilder.JointDofConfig(
        axis=newton.Axis.Y,
        actuator_mode=newton.JointTargetMode.NONE,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    effort_joint_1 = newton.ModelBuilder.JointDofConfig(
        axis=newton.Axis.Y,
        actuator_mode=newton.JointTargetMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
    )
    effort_joint_other = newton.ModelBuilder.JointDofConfig(
        axis=newton.Axis.Y,
        actuator_mode=newton.JointTargetMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    pd_joint_dof_config = newton.ModelBuilder.JointDofConfig(
        axis=newton.Axis.Y,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
        target_ke=1000.0,
        target_kd=20.0,
        limit_lower=qmin,
        limit_upper=qmax,
    )

    # Add a revolute joint between link 1 and link 2
    if fixed:
        j1 = _builder.add_joint_fixed(
            label="link1_to_link2",
            parent=bid1,
            child=bid2,
            parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
        )
        j2 = _builder.add_joint_fixed(
            label="link2_to_link3",
            parent=bid2,
            child=bid3,
            parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
        )
        j3 = _builder.add_joint_fixed(
            label="link3_to_link4",
            parent=bid3,
            child=bid4,
            parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
        )
        _builder.add_joint_fixed(
            label="link4_to_link1",
            parent=bid4,
            child=bid1,
            parent_xform=wp.transformf(r_j4 - r_b4, wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(r_j4 - r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    else:
        joint_1_axis = (
            pd_joint_dof_config
            if implicit_pd and 1 in actuator_ids
            else effort_joint_1
            if 1 in actuator_ids
            else passive_joint_dof_config
        )
        j1 = _builder.add_joint_revolute(
            label="link1_to_link2",
            parent=bid1,
            child=bid2,
            axis=joint_1_axis,
            parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
        )

        if overconstrained:
            # Add a revolute joint between link 2 and link 3
            j2 = _builder.add_joint_revolute(
                label="link2_to_link3",
                parent=bid2,
                child=bid3,
                axis=effort_joint_other if 2 in actuator_ids else passive_joint_dof_config,
                parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
                child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
            )

            # Add a revolute joint between link 3 and link 4
            j3 = _builder.add_joint_revolute(
                label="link3_to_link4",
                parent=bid3,
                child=bid4,
                axis=effort_joint_other if 3 in actuator_ids else passive_joint_dof_config,
                parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
                child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
            )
        else:
            # Add a 2-DoF rotational joint (universal) between link 2 and link 3
            j2_axes = [
                newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.Y,
                    actuator_mode=newton.JointTargetMode.NONE,
                    limit_lower=qmin,
                    limit_upper=qmax,
                ),
                newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.Z,
                    actuator_mode=newton.JointTargetMode.NONE,
                ),
            ]
            j2 = _builder.add_joint_d6(
                label="link2_to_link3",
                parent=bid2,
                child=bid3,
                parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
                child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
                angular_axes=j2_axes,
            )

            # Add a spherical joint between link 3 and link 4
            j3 = _builder.add_joint_ball(
                label="link3_to_link4",
                parent=bid3,
                child=bid4,
                parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
                child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
            )

        # Add a revolute joint between link 4 and link 1 (closes the loop)
        _builder.add_joint_revolute(
            label="link4_to_link1",
            parent=bid4,
            child=bid1,
            axis=effort_joint_other if 4 in actuator_ids else passive_joint_dof_config,
            parent_xform=wp.transformf(r_j4 - r_b4, wp.quat_identity(dtype=wp.float32)),
            child_xform=wp.transformf(r_j4 - r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    # Add the joints to the articulation
    articulation_joints.extend([j1, j2, j3])
    _builder.add_articulation(articulation_joints)

    _builder.end_world()

    # Return the populated model builder
    return _builder


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / 0.0025))

        self.frame_dt = self.frame_dt / self.sim_substeps
        self.sim_substeps = 1

        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_frame = 0
        self.world_count = args.world_count if args else 1
        self.viewer = viewer
        self.device = wp.get_device()
        self.graph_capture = args.graph_capture
        self.show_solver_status = args.show_solver_status
        self.plot_solver_info = args.plot_solver_info

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0

        # Manually build the basic four-bar mechanism using the builder API
        build_boxes_fourbar(
            builder=robot_builder,
            floatingbase=True,
            dynamic_joints=False,
            implicit_pd=False,
            limits=False,
            overconstrained=args.overconstrained,
            fixed=args.fixed_model,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.gravity.fill_(0.0)

        # Set initial velocities
        lin_vel_global = np.array([0.0, 0.0, 0.0])
        ang_vel_global = np.array([1.0, 1.0, 1.0]) * 10.0
        body_q_np = self.model.body_q.numpy()
        body_com_np = self.model.body_com.numpy()
        body_mass_np = self.model.body_mass.numpy()
        com = np.zeros((3,))
        mass = 0.0
        for body_id in range(self.model.body_count):
            com += body_mass_np[body_id] * (body_q_np[body_id, 0:3] + body_com_np[body_id, :])
            mass += body_mass_np[body_id]
        com = com / mass
        body_qd = []
        for body_id in range(self.model.body_count):
            lin_vel = lin_vel_global + np.cross(ang_vel_global, body_q_np[body_id, 0:3] - com)
            body_qd.append(np.concatenate((lin_vel, ang_vel_global)))
        self.model.body_qd.assign(body_qd)

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.collect_solver_info = self.plot_solver_info
        # solver_config.sparse_dynamics = True
        # solver_config.sparse_jacobian = True
        # solver_config.dynamics.linear_solver_type = "CR"
        # solver_config.dynamics.linear_solver_kwargs = {"maxiter": 9}
        solver_config.dynamics.linear_solver_type = "LLTS"
        solver_config.use_collision_detector = False
        solver_config.use_fk_solver = True
        solver_config.collision_detector.pipeline = "primitive"
        solver_config.collision_detector.max_contacts = 32 * self.model.world_count
        solver_config.dynamics.preconditioning = True
        solver_config.padmm.primal_tolerance = 1e-4
        solver_config.padmm.dual_tolerance = 1e-4
        solver_config.padmm.compl_tolerance = 1e-4
        solver_config.padmm.max_iterations = 200
        solver_config.padmm.rho_0 = 0.1
        # solver_config.padmm.use_acceleration = True
        solver_config.padmm.use_acceleration = False
        # solver_config.padmm.warmstart_mode = "containers"
        solver_config.padmm.warmstart_mode = "none"
        solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"
        # solver_config.padmm.use_graph_conditionals = False

        # Create the Kamino solver for the given model
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        # Create state, control, and contacts data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Warm-start the simulation
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.reset(self.state_0)

        self.state_0.body_qd.assign(self.model.body_qd)
        # Updating joint velocities from body velocities
        newton.eval_ik(
            model=self.model,
            state=self.state_0,
            joint_q=self.state_0.joint_q,
            joint_qd=self.state_0.joint_qd,
        )

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(-0.5, -1.0, 0.2)
            pitch = -5.0
            yaw = 70.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda and self.graph_capture:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            # self.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_0)

            if self.sim_substeps % 2 == 1 and i == self.sim_substeps - 1:
                # Swap states by copying the state arrays for graph capture
                self.state_0.assign(self.state_1)
            else:
                # We can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # print(f"Frame {self.sim_frame}")
        # if self.sim_frame >= 5:
        # exit()

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.sim_frame += 1

        if self.show_solver_status:
            status = self.solver._solver_kamino.solver_fd._data.status.numpy()
            iterations = status[0][1]
            r_primal = status[0][2]
            r_dual = status[0][3]
            r_compl = status[0][4]
            print("ADMM iterations: ", iterations)
            self.viewer.log_scalar("ADMM iterations", iterations)
            self.viewer.log_scalar("Primal residual", r_primal)
            self.viewer.log_scalar("Dual residual", r_dual)
            self.viewer.log_scalar("Complementarity residual", r_compl)

        if self.plot_solver_info:

            def save_solver_info(solver: PADMMSolver, path: str | None = None, verbose: bool = False):
                # Attempt to import matplotlib for plotting
                try:
                    import matplotlib.pyplot as plt  # noqa: PLC0415
                except ImportError:
                    return  # matplotlib is not available so we skip plotting

                solver_has_acceleration = solver._use_acceleration

                nw = solver.size.num_worlds
                status = solver.data.status.numpy()
                iterations = [status[w][1] for w in range(nw)]
                offsets_np = solver.data.info.offsets.numpy()
                num_rho_updates_np = extract_info_vectors(
                    offsets_np, solver.data.info.num_rho_updates.numpy(), iterations
                )
                norm_s_np = extract_info_vectors(offsets_np, solver.data.info.norm_s.numpy(), iterations)
                norm_x_np = extract_info_vectors(offsets_np, solver.data.info.norm_x.numpy(), iterations)
                norm_y_np = extract_info_vectors(offsets_np, solver.data.info.norm_y.numpy(), iterations)
                norm_z_np = extract_info_vectors(offsets_np, solver.data.info.norm_z.numpy(), iterations)
                f_ccp_np = extract_info_vectors(offsets_np, solver.data.info.f_ccp.numpy(), iterations)
                f_ncp_np = extract_info_vectors(offsets_np, solver.data.info.f_ncp.numpy(), iterations)
                r_dx_np = extract_info_vectors(offsets_np, solver.data.info.r_dx.numpy(), iterations)
                r_dy_np = extract_info_vectors(offsets_np, solver.data.info.r_dy.numpy(), iterations)
                r_dz_np = extract_info_vectors(offsets_np, solver.data.info.r_dz.numpy(), iterations)
                r_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_primal.numpy(), iterations)
                r_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_dual.numpy(), iterations)
                r_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_compl.numpy(), iterations)
                r_pd_np = extract_info_vectors(offsets_np, solver.data.info.r_pd.numpy(), iterations)
                r_dp_np = extract_info_vectors(offsets_np, solver.data.info.r_dp.numpy(), iterations)
                r_ncp_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_primal.numpy(), iterations)
                r_ncp_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_dual.numpy(), iterations)
                r_ncp_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_compl.numpy(), iterations)
                r_ncp_natmap_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_natmap.numpy(), iterations)

                if solver_has_acceleration:
                    num_restarts_np = extract_info_vectors(
                        offsets_np,
                        solver.data.info.num_restarts.numpy(),
                        iterations,
                    )
                    a_np = extract_info_vectors(offsets_np, solver.data.info.a.numpy(), iterations)
                    r_comb_np = extract_info_vectors(offsets_np, solver.data.info.r_comb.numpy(), iterations)
                    r_comb_ratio_np = extract_info_vectors(
                        offsets_np,
                        solver.data.info.r_comb_ratio.numpy(),
                        iterations,
                    )

                # List of (label, data) for plotting
                info_list = [
                    ("num_rho_updates", num_rho_updates_np),
                    ("norm_s", norm_s_np),
                    ("norm_x", norm_x_np),
                    ("norm_y", norm_y_np),
                    ("norm_z", norm_z_np),
                    ("f_ccp", f_ccp_np),
                    ("f_ncp", f_ncp_np),
                    ("r_dx", r_dx_np),
                    ("r_dy", r_dy_np),
                    ("r_dz", r_dz_np),
                    ("r_primal", r_primal_np),
                    ("r_dual", r_dual_np),
                    ("r_compl", r_compl_np),
                    ("r_pd", r_pd_np),
                    ("r_dp", r_dp_np),
                    ("r_ncp_primal", r_ncp_primal_np),
                    ("r_ncp_dual", r_ncp_dual_np),
                    ("r_ncp_compl", r_ncp_compl_np),
                    ("r_ncp_natmap", r_ncp_natmap_np),
                ]
                if solver_has_acceleration:
                    info_list.extend(
                        [
                            ("num_restarts", num_restarts_np),
                            ("a", a_np),
                            ("r_comb", r_comb_np),
                            ("r_comb_ratio", r_comb_ratio_np),
                        ]
                    )

                # Plot all info as subplots: rows=info_list, cols=worlds
                n_rows = len(info_list)
                n_cols = nw
                _fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(4 * n_cols, 2.5 * n_rows),
                    squeeze=False,
                )
                for row, (label, arr) in enumerate(info_list):
                    for col in range(nw):
                        ax = axes[row, col]
                        ax.plot(arr[col], label=f"{label}")
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel(label)
                        if row == 0:
                            ax.set_title(f"World {col}")
                        if col == 0:
                            ax.set_ylabel(label)
                        else:
                            ax.set_ylabel("")
                        ax.grid(True)
                plt.tight_layout()
                if path is not None:
                    plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
                plt.close()

            plot_filename = Path(os.path.dirname(__file__)) / "plots" / f"padmm_plot_{self.sim_frame}.pdf"
            plot_filename.parent.mkdir(parents=True, exist_ok=True)
            save_solver_info(
                self.solver._solver_kamino._solver_fd,
                str(plot_filename),
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        # Since rendering is called after stepping the simulation, the previous and next
        # states correspond to self.state_1 and self.state_0 due to the reference swaps,
        # so contacts are rendered with self.state_1 to match the body positions at the
        # time of contact generation.
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    def apply_forces(self, state: newton.State):

        @wp.kernel
        def force_application_kernel(
            body_q: wp.array[wp.transform],
            body_f: wp.array[wp.spatial_vector],
            body_count: int,
            body_com: wp.array[wp.vec3f],
            body_mass: wp.array[wp.float32],
            body_inertia: wp.array[wp.mat33],
        ):
            # body_id = wp.tid()

            com = wp.vec3f(0.0)
            mass = wp.float32(0.0)
            for body_id in range(body_count):
                pos = wp.transform_get_translation(body_q[body_id])
                rot = wp.transform_get_rotation(body_q[body_id])
                com += (pos + wp.quat_rotate(rot, body_com[body_id])) * body_mass[body_id]
                mass += body_mass[body_id]
            com = com / mass

            target_lin_acc = wp.vec3f(0.01, 0.01, 0.01)
            target_ang_acc = wp.vec3f(1.0, 1.0, 1.0)

            for body_id in range(body_count):
                pos = wp.transform_get_translation(body_q[body_id])
                rot = wp.transform_get_rotation(body_q[body_id])

                offset = (pos + wp.quat_rotate(rot, body_com[body_id])) - com
                target_lin_acc += wp.cross(target_ang_acc, offset)

                rot = wp.transform_get_rotation(body_q[body_id])

                body_force = target_lin_acc * body_mass[body_id]
                body_torque = wp.quat_rotate(rot, body_inertia[body_id] * wp.quat_rotate_inv(rot, target_ang_acc))

                body_f[body_id] += wp.spatial_vector(body_force, body_torque)

        wp.launch(
            kernel=force_application_kernel,
            # dim=self.model.body_count,
            dim=1,
            inputs=[
                state.body_q,
                state.body_f,
                self.model.body_count,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_inertia,
            ],
            device=self.model.device,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        parser.add_argument(
            "--overconstrained",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the overconstrained version of the fourbar model.",
        )
        parser.add_argument(
            "--fixed-model",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether the model should be made up of fixed joints.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use graph-capture.",
        )
        parser.add_argument(
            "--show-solver-status",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Visualizes the solver status with interactive plots.",
        )
        parser.add_argument(
            "--plot-solver-info",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Plots the solver information after every step.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
