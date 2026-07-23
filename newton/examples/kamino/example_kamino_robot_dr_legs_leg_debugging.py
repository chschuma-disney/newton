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

import os
from pathlib import Path

import warp as wp

import newton
import newton.examples


def get_builder() -> newton.ModelBuilder:
    robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
    robot_builder.default_shape_cfg.margin = 1e-6
    robot_builder.default_shape_cfg.gap = 1e-2

    asset_filepath = str(
        (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / ".."
            / ".."
            / ".."
            / ".."
            / "kamino-assets-disney"
            / "usda"
            / "dr_legs_leg"
            / "dr_legs_leg.usda"
        ).resolve()
    )
    robot_builder.add_usd(
        asset_filepath,
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )

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
        self.linear_solver_type = getattr(args, "linear_solver_type", "LLTB") if args else "LLTB"
        self.linear_solver_kwargs = getattr(args, "linear_solver_kwargs", {}) if args else {}
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder
        robot_builder = get_builder()

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
        self.config.use_collision_detector = False
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
        self.contacts = None

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Warm-start the simulation
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.solver.reset(self.state_0)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 0.4), q_b)
        self.base_q.assign([q_base] * self.world_count)
        reset_config = newton.solvers.SolverKamino.ResetConfig(
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(base_q=self.base_q),
        )
        self.solver.reset(state=self.state_0, config=reset_config)

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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
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
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer.gui._cam_speed = 0.5
    newton.examples.run(Example(viewer, args), args)
