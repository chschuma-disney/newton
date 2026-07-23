# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot DR Legs
#
# Shows how to simulate DR Legs with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_robot_dr_legs --world-count 16
#
###########################################################################

import argparse
import os
from pathlib import Path
from typing import Literal

import warp as wp

import newton
import newton.examples


def get_asset_filepath(model_version: Literal["v1", "v2", "v3"]) -> str:
    if model_version == "v1":
        asset_path = newton.utils.download_asset("disneyresearch", ref="d69b2e04bc1fc246c415f0549e5f02d8aae1ef31")
        return str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")
    elif model_version == "v2":
        asset_path = newton.utils.download_asset("disneyresearch", ref="4a44e9642d9650f38f0126cf6cc269a50ff7c5e3")
        return str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")
    elif model_version == "v3":
        return str(Path(os.path.dirname(os.path.abspath(__file__))) / "models" / "dr_legs_with_meshes_and_boxes.usda")


def get_builder(model_version: Literal["v1", "v2", "v3", "v2_converted"]) -> newton.ModelBuilder:
    robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
    robot_builder.default_shape_cfg.margin = 1e-6
    robot_builder.default_shape_cfg.gap = 1e-2

    model_version_import = "v2" if model_version == "v2_converted" else model_version

    asset_filepath = get_asset_filepath(model_version_import)
    robot_builder.add_usd(
        asset_filepath,
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )

    if model_version == "v2_converted":
        asset_filepath_old = get_asset_filepath("v1")
        robot_builder_old = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder_old)
        robot_builder_old.add_usd(
            asset_filepath_old,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        joint_names = [label.split("/")[-1] for label in robot_builder.joint_label]
        joint_names_old = [label.split("/")[-1] for label in robot_builder_old.joint_label]

        joint_offsets = {
            "j6_l_i": -0.01,
            "j6_l_o": 0.01,
            "j6_r_i": -0.01,
            "j6_r_o": 0.01,
        }

        for joint_id in range(len(robot_builder.joint_parent)):
            if robot_builder.joint_parent[joint_id] == -1:
                continue
            joint_name = joint_names[joint_id]
            joint_id_old = joint_names_old.index(joint_name)
            joint_pos = (
                robot_builder_old.body_q[robot_builder_old.joint_parent[joint_id_old]][:3]
                + robot_builder_old.joint_X_p[joint_id_old][:3]
            )
            if joint_name in joint_offsets:
                rot = wp.transform_get_rotation(robot_builder.joint_X_p[joint_id])
                joint_pos += wp.quat_rotate(rot, wp.vec3f(1.0, 0.0, 0.0)) * joint_offsets[joint_name]
            joint_X_p = robot_builder.joint_X_p[joint_id]
            joint_X_p[:3] = joint_pos - robot_builder.body_q[robot_builder.joint_parent[joint_id]][:3]
            robot_builder.joint_X_p[joint_id] = joint_X_p
            joint_X_c = robot_builder.joint_X_c[joint_id]
            joint_X_c[:3] = joint_pos - robot_builder.body_q[robot_builder.joint_child[joint_id]][:3]
            robot_builder.joint_X_c[joint_id] = joint_X_c

            robot_builder.joint_axis[joint_id] = robot_builder_old.joint_axis[joint_id_old]

    # Print joint axes
    # for joint_id in range(1, len(robot_builder.joint_parent)):
    #     rot = wp.transform_get_rotation(robot_builder.joint_X_c[joint_id])
    #     axis = wp.quat_rotate(rot, robot_builder.joint_axis[5 + joint_id])
    #     print(robot_builder.joint_label[joint_id], axis, robot_builder.joint_axis[5 + joint_id])

    body_labels = [label.split("/")[-1] for label in robot_builder.body_label]

    foot_l_id = body_labels.index("foot_l")
    foot_r_id = body_labels.index("foot_r")
    # pelvis_id = body_labels.index("pelvis")

    robot_builder.add_joint_fixed(
        -1,
        foot_l_id,
        parent_xform=robot_builder.body_q[foot_l_id],
        child_xform=wp.transform_identity(),
    )
    robot_builder.add_joint_fixed(
        -1,
        foot_r_id,
        parent_xform=robot_builder.body_q[foot_r_id],
        child_xform=wp.transform_identity(),
    )
    # robot_builder.add_joint_prismatic(
    #     -1,
    #     pelvis_id,
    #     parent_xform=robot_builder.body_q[pelvis_id],
    #     child_xform=wp.transform_identity(),
    #     axis=wp.vec3f(0.0, 0.0, 1.0),
    # )

    return robot_builder


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / 0.01))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.use_kamino_contacts = args.use_kamino_contacts if args else False
        self.linear_solver_type = getattr(args, "linear_solver_type", "LLTB") if args else "LLTB"
        self.linear_solver_kwargs = getattr(args, "linear_solver_kwargs", {}) if args else {}
        self.viewer = viewer
        self.device = wp.get_device()
        self.show_solver_status = args.show_solver_status

        # Create a single-robot model builder
        robot_builder = get_builder(args.model)

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.request_contact_attributes("force")
        builder.default_shape_cfg.margin = 1e-6
        builder.default_shape_cfg.gap = 1e-2
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        # builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 72 * self.world_count

        if args.passive:
            self.model.joint_armature.zero_()
            self.model.joint_damping.zero_()
            self.model.joint_target_kd.zero_()
            self.model.joint_target_ke.zero_()

        else:
            joint_mask = self.model.joint_target_mode.numpy() != newton.JointTargetMode.NONE

            joint_armature = self.model.joint_armature.numpy()
            joint_armature[joint_mask] = 0.011
            self.model.joint_armature.assign(joint_armature)
            joint_damping = self.model.joint_damping.numpy()
            joint_damping[joint_mask] = 0.044
            self.model.joint_damping.assign(joint_damping)
            joint_target_ke = self.model.joint_target_ke.numpy()
            joint_target_ke[joint_mask] = 10.0
            self.model.joint_target_ke.assign(joint_target_ke)
            joint_target_kd = self.model.joint_target_kd.numpy()
            joint_target_kd[joint_mask] = 2.0
            self.model.joint_armature.assign(joint_target_kd)

        # self.model.shape_gap.fill_(0.01)
        # print(self.model.shape_gap)

        # self.model.gravity.fill_(0.0)

        # Create the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(self.model)
        self.config.use_fk_solver = True
        self.config.use_collision_detector = self.use_kamino_contacts
        self.config.dynamics.linear_solver_type = self.linear_solver_type
        self.config.dynamics.linear_solver_kwargs = self.linear_solver_kwargs
        # self.config.constraints.delta = 1e-3
        self.config.padmm.max_iterations = 200
        # self.config.padmm.primal_tolerance = 1e-4
        # self.config.padmm.dual_tolerance = 1e-4
        # self.config.padmm.compl_tolerance = 1e-4
        self.config.padmm.rho_0 = 0.5
        self.config.padmm.use_graph_conditionals = getattr(args, "use_graph_conditionals", True) if args else True
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Configure CD components based on whether we want to use Kamino's
        # internal contact solver or Newton's collision pipeline
        if not self.use_kamino_contacts:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)
        else:
            self.collision_pipeline = None
            self.contacts = self.model.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Warm-start the simulation
        if not self.use_kamino_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.solver.reset(self.state_0)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.graph = None
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(1.34, 0.0, 0.25)
            pitch = -7.0
            yaw = -180.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if not self.use_kamino_contacts:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            else:
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        if self.show_solver_status:
            status = self.solver._solver_kamino.solver_fd._data.status.numpy()
            iterations = status[0][1]
            r_primal = status[0][2]
            r_dual = status[0][3]
            r_compl = status[0][4]
            self.viewer.log_scalar("ADMM iterations", iterations)
            self.viewer.log_scalar("Primal residual", r_primal)
            self.viewer.log_scalar("Dual residual", r_dual)
            self.viewer.log_scalar("Complementarity residual", r_compl)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.add_argument(
            "--linear-solver-type",
            choices=("LLTB", "LLTBRCM", "CR"),
            default="LLTB",
            type=str.upper,
            help="Kamino dynamics linear solver to use.",
        )
        parser.add_argument(
            "--no-graph-conditionals",
            dest="use_graph_conditionals",
            action="store_false",
            help="Disable CUDA graph conditional nodes in Kamino PADMM.",
        )
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=False)
        parser.add_argument(
            "--model",
            choices=("v1", "v2", "v3", "v2_converted"),
            default="v2",
            help="Model variant to use.",
        )
        parser.add_argument(
            "--passive",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Whether the model should be passive (no PD control).",
        )
        parser.add_argument(
            "--show-solver-status",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Visualizes the solver status with interactive plots.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer.gui._cam_speed = 0.5
    newton.examples.run(Example(viewer, args), args)
