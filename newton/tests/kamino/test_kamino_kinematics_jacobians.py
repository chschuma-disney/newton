# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for `kinematics/jacobians.py`.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino._src.core.math import compute_body_pose_update_with_logmap, quat_exp
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.constraints import make_unilateral_constraints_info
from newton._src.solvers.kamino._src.kinematics.jacobians import (
    ColMajorSparseConstraintJacobians,
    ConstraintJacobianMethod,
    DenseSystemJacobians,
    SparseSystemJacobians,
)
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.models.builders.basics import (
    build_boxes_fourbar,
    make_basics_heterogeneous_builder,
)
from newton._src.solvers.kamino._src.models.builders.testing import (
    build_binary_revolute_joint_test,
    build_binary_universal_joint_test,
)
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests import setup_tests, test_context
from newton.tests.utils.extract import extract_cts_jacobians, extract_dofs_jacobians
from newton.tests.utils.make import (
    make_test_problem,
    make_test_problem_fourbar,
    make_test_problem_heterogeneous,
)

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _integrate_body_poses_logmap(
    poses_in: wp.array[wp.transformf],
    twists: wp.array[wp.spatial_vectorf],
    dt: wp.float32,
    poses_out: wp.array[wp.transformf],
):
    """Integrate body poses with the same log-map update used by :class:`IntegratorEuler`."""
    i = wp.tid()
    u_i = twists[i]
    poses_out[i] = compute_body_pose_update_with_logmap(
        dt,
        poses_in[i],
        wp.spatial_top(u_i),
        wp.spatial_bottom(u_i),
    )


###
# Tests
###


class TestKinematicsDenseSystemJacobians(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_allocate_single_dense_system_jacobians_only_joints(self):
        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_bilateral_joint_cts: {model.size.sum_of_num_bilateral_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model)
        if self.verbose:
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = model.size.sum_of_num_bilateral_joint_cts + model.size.sum_of_num_bounded_joint_cts
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_02_allocate_single_dense_system_jacobians_with_limits(self):
        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_bilateral_joint_cts: {model.size.sum_of_num_bilateral_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = (
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + limits.model_max_limits_host
        )
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_03_allocate_single_dense_system_jacobians_with_contacts(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_bilateral_joint_cts: {model.size.sum_of_num_bilateral_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, contacts=contacts)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = (
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + 3 * contacts.model_max_contacts_host
        )
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_04_allocate_single_dense_system_jacobians_with_limits_and_contacts(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = build_boxes_fourbar()

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_bilateral_joint_cts: {model.size.sum_of_num_bilateral_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Check the allocations of Jacobians
        model_num_cts = (
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + limits.model_max_limits_host
            + 3 * contacts.model_max_contacts_host
        )
        self.assertEqual(jacobians.data.J_dofs_offsets.size, 1)
        self.assertEqual(jacobians.data.J_cts_offsets.size, 1)
        self.assertEqual(jacobians.data.J_dofs_offsets.numpy()[0], 0)
        self.assertEqual(jacobians.data.J_cts_offsets.numpy()[0], 0)
        self.assertEqual(
            jacobians.data.J_dofs_data.shape, (model.size.sum_of_num_joint_dofs * model.size.sum_of_num_body_dofs,)
        )
        self.assertEqual(jacobians.data.J_cts_data.shape, (model_num_cts * model.size.sum_of_num_body_dofs,))

    def test_05_allocate_homogeneous_dense_system_jacobians(self):
        # Problem constants
        num_worlds = 3
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_fourbar)

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_bilateral_joint_cts: {model.size.sum_of_num_bilateral_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Build model info
        make_unilateral_constraints_info(model, model.data(), limits, contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Compute the total maximum number of constraints
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        if self.verbose:
            print("num_body_dofs: ", num_body_dofs)
            print("max_total_cts: ", max_total_cts)
            print("num_joint_dofs: ", num_joint_dofs)

        # Compute Jacobian sizes
        J_dofs_size: list[int] = [0] * num_worlds
        J_cts_size: list[int] = [0] * num_worlds
        for w in range(num_worlds):
            J_dofs_size[w] = num_joint_dofs[w] * num_body_dofs[w]
            J_cts_size[w] = max_total_cts[w] * num_body_dofs[w]

        # Compute Jacobian offsets
        J_dofs_offsets: list[int] = [0] + [sum(J_dofs_size[:w]) for w in range(1, num_worlds)]
        J_cts_offsets: list[int] = [0] + [sum(J_cts_size[:w]) for w in range(1, num_worlds)]

        # Check the allocations of Jacobians
        self.assertEqual(jacobians.data.J_dofs_offsets.size, num_worlds)
        self.assertEqual(jacobians.data.J_cts_offsets.size, num_worlds)
        J_dofs_mio_np = jacobians.data.J_dofs_offsets.numpy()
        J_cts_mio_np = jacobians.data.J_cts_offsets.numpy()
        for w in range(num_worlds):
            self.assertEqual(J_dofs_mio_np[w], J_dofs_offsets[w])
            self.assertEqual(J_cts_mio_np[w], J_cts_offsets[w])
        self.assertEqual(jacobians.data.J_dofs_data.size, sum(J_dofs_size))
        self.assertEqual(jacobians.data.J_cts_data.size, sum(J_cts_size))

    def test_06_allocate_heterogeneous_dense_system_jacobians(self):
        # Problem constants
        max_world_contacts = 12

        # Construct the model description using the ModelBuilderKamino
        builder = make_basics_heterogeneous_builder()
        num_worlds = builder.num_worlds

        # Create the model from the builder
        model = builder.finalize(device=self.default_device)
        if self.verbose:
            print("")  # Add a newline for better readability
            print(f"model.size.sum_of_num_bodies: {model.size.sum_of_num_bodies}")
            print(f"model.size.sum_of_num_joints: {model.size.sum_of_num_joints}")
            print(f"model.size.sum_of_num_bilateral_joint_cts: {model.size.sum_of_num_bilateral_joint_cts}")
            print(f"model.size.sum_of_num_joint_dofs: {model.size.sum_of_num_joint_dofs}")

        # Construct and allocate the limits container
        limits = LimitsKamino(model=model)
        if self.verbose:
            print("limits.model_max_limits_host: ", limits.model_max_limits_host)
            print("limits.world_max_limits_host: ", limits.world_max_limits_host)

        # Set the contact allocation capacities
        required_world_max_contacts = [max_world_contacts] * builder.num_worlds
        if self.verbose:
            print("required_world_max_contacts: ", required_world_max_contacts)

        # Construct and allocate the contacts container
        contacts = ContactsKamino(capacity=required_world_max_contacts, device=self.default_device)
        if self.verbose:
            print("contacts.default_max_world_contacts: ", contacts.default_max_world_contacts)
            print("contacts.model_max_contacts_host: ", contacts.model_max_contacts_host)
            print("contacts.world_max_contacts_host: ", contacts.world_max_contacts_host)

        # Build model info
        make_unilateral_constraints_info(model, model.data(), limits, contacts)

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_dofs_data: shape={jacobians.data.J_dofs_data.shape}")
            print(f"J_cts_data: shape={jacobians.data.J_cts_data.shape}")

        # Compute the total maximum number of constraints
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        if self.verbose:
            print("num_body_dofs: ", num_body_dofs)
            print("max_total_cts: ", max_total_cts)
            print("num_joint_dofs: ", num_joint_dofs)

        # Compute Jacobian sizes
        J_dofs_size: list[int] = [0] * num_worlds
        J_cts_size: list[int] = [0] * num_worlds
        for w in range(num_worlds):
            J_dofs_size[w] = num_joint_dofs[w] * num_body_dofs[w]
            J_cts_size[w] = max_total_cts[w] * num_body_dofs[w]

        # Compute Jacobian offsets
        J_dofs_offsets: list[int] = [0] + [sum(J_dofs_size[:w]) for w in range(1, num_worlds)]
        J_cts_offsets: list[int] = [0] + [sum(J_cts_size[:w]) for w in range(1, num_worlds)]

        # Check the allocations of Jacobians
        self.assertEqual(jacobians.data.J_dofs_offsets.size, num_worlds)
        self.assertEqual(jacobians.data.J_cts_offsets.size, num_worlds)
        J_dofs_mio_np = jacobians.data.J_dofs_offsets.numpy()
        J_cts_mio_np = jacobians.data.J_cts_offsets.numpy()
        for w in range(num_worlds):
            self.assertEqual(J_dofs_mio_np[w], J_dofs_offsets[w])
            self.assertEqual(J_cts_mio_np[w], J_cts_offsets[w])
        self.assertEqual(jacobians.data.J_dofs_data.size, sum(J_dofs_size))
        self.assertEqual(jacobians.data.J_cts_data.size, sum(J_cts_size))

    def test_07_build_single_dense_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Reshape the flat actuation Jacobian as a matrix
        J_dofs_offsets = jacobians.data.J_dofs_offsets.numpy()
        J_dofs_flat = jacobians.data.J_dofs_data.numpy()
        njd = J_dofs_flat.size // model.size.sum_of_num_body_dofs
        J_dofs_mat = J_dofs_flat.reshape((njd, model.size.sum_of_num_body_dofs))

        # Reshape the flat constraintJacobian as a matrix
        J_cts_offsets = jacobians.data.J_cts_offsets.numpy()
        J_cts_flat = jacobians.data.J_cts_data.numpy()
        maxncts = J_cts_flat.size // model.size.sum_of_num_body_dofs
        J_cts_mat = J_cts_flat.reshape((maxncts, model.size.sum_of_num_body_dofs))

        # Check the shapes of the Jacobians
        self.assertEqual(J_dofs_offsets.size, 1)
        self.assertEqual(J_cts_offsets.size, 1)
        self.assertEqual(
            maxncts,
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + limits.model_max_limits_host
            + 3 * contacts.model_max_contacts_host,
        )
        self.assertEqual(njd, model.size.sum_of_num_joint_dofs)

        # Optional verbose output
        if self.verbose:
            print(f"J_cts_offsets (shape={jacobians.data.J_cts_offsets.shape}): {jacobians.data.J_cts_offsets}")
            print(f"J_cts_flat (shape={J_cts_flat.shape}):\n{J_cts_flat}")
            print(f"J_cts_mat (shape={J_cts_mat.shape}):\n{J_cts_mat}")
            print(f"J_dofs_offsets (shape={jacobians.data.J_dofs_offsets.shape}): {jacobians.data.J_dofs_offsets}")
            print(f"J_dofs_flat (shape={J_dofs_flat.shape}):\n{J_dofs_flat}")
            print(f"J_dofs_mat (shape={J_dofs_mat.shape}):\n{J_dofs_mat}")

    def test_08_build_homogeneous_dense_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Extract the Jacobian matrices
        J_cts = extract_cts_jacobians(model=model, limits=limits, contacts=contacts, jacobians=jacobians)
        J_dofs = extract_dofs_jacobians(model=model, jacobians=jacobians)
        for w in range(model.size.num_worlds):
            msg.info("[world='%d']: J_cts:\n%s", w, J_cts[w])
            msg.info("[world='%d']: J_dofs:\n%s", w, J_dofs[w])

    def test_09_build_heterogeneous_dense_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the dense system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Extract the Jacobian matrices
        J_cts = extract_cts_jacobians(model=model, limits=limits, contacts=contacts, jacobians=jacobians)
        J_dofs = extract_dofs_jacobians(model=model, jacobians=jacobians)
        for w in range(model.size.num_worlds):
            msg.info("[world='%d']: J_cts:\n%s", w, J_cts[w])
            msg.info("[world='%d']: J_dofs:\n%s", w, J_dofs[w])


class TestKinematicsSparseSystemJacobians(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.epsilon = 1e-6  # Threshold for sparse-dense comparison test

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    ###
    # Helpers
    ###

    def _compare_dense_sparse_jacobians(
        self,
        model: ModelKamino,
        limits: LimitsKamino | None,
        contacts: ContactsKamino | None,
        jacobians_dense: DenseSystemJacobians,
        jacobians_sparse: SparseSystemJacobians,
    ):
        # Reshape the dense Jacobian data as a matrices
        J_cts_dense = extract_cts_jacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians_dense, verbose=self.verbose
        )
        J_dofs_dense = extract_dofs_jacobians(model=model, jacobians=jacobians_dense, verbose=self.verbose)

        # Get the (dense) numpy version of the sparse Jacobians
        J_dofs_sparse = jacobians_sparse._J_dofs.bsm.numpy()
        J_cts_sparse = jacobians_sparse._J_cts.bsm.numpy()

        self.assertEqual(len(J_cts_dense), len(J_cts_sparse))
        self.assertEqual(len(J_dofs_dense), len(J_dofs_sparse))

        # Check that Jacobians match
        for mat_id in range(len(J_cts_dense)):
            if J_dofs_dense[mat_id].size > 0:
                diff_J_dofs = J_dofs_dense[mat_id] - J_dofs_sparse[mat_id]
                self.assertLess(np.max(np.abs(diff_J_dofs)), self.epsilon)

            diff_J_cts = J_cts_dense[mat_id][: J_cts_sparse[mat_id].shape[0], :] - J_cts_sparse[mat_id]
            self.assertLess(np.max(np.abs(diff_J_cts)), self.epsilon)

            # Extra entries in dense constraint Jacobian need to be zero
            if J_cts_dense[mat_id].shape[0] > J_cts_sparse[mat_id].shape[0]:
                self.assertEqual(np.max(np.abs(J_cts_dense[mat_id][J_cts_sparse[mat_id].shape[0] :, :])), 0)

    def _compare_row_col_major_jacobians(
        self,
        jacobians: SparseSystemJacobians,
        jacobians_col_major: ColMajorSparseConstraintJacobians,
    ):
        # Get the (dense) numpy version of the Jacobians
        J_cts_row_major = jacobians._J_cts.bsm.numpy()
        J_cts_col_major = jacobians_col_major.bsm.numpy()

        self.assertEqual(len(J_cts_row_major), len(J_cts_col_major))

        # Check that Jacobians match
        for mat_id in range(len(J_cts_row_major)):
            diff_J_cts = J_cts_row_major[mat_id] - J_cts_col_major[mat_id]
            max_diff = np.max(np.abs(diff_J_cts))
            if max_diff > self.epsilon and self.verbose:
                msg.warning(f"[{mat_id}] J_cts_row_major:\n{J_cts_row_major[mat_id]}")
                msg.warning(f"[{mat_id}] J_cts_col_major:\n{J_cts_col_major[mat_id]}")
            self.assertLess(max_diff, self.epsilon)

    ###
    # Construction
    ###

    def test_01_allocate_single_sparse_system_jacobians_only_joints(self):
        # Construct the test problem
        model, *_ = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the sparse Jacobians
        jacobians = SparseSystemJacobians(model=model)
        self.assertIs(jacobians._J_cts.bsm.device, model.device)
        self.assertIs(jacobians._J_cts.device, model.device)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = model.size.sum_of_num_bilateral_joint_cts + model.size.sum_of_num_bounded_joint_cts
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_02_allocate_single_sparse_system_jacobians_with_limits(self):
        # Construct the test problem
        model, _data, _state, limits, *_ = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=True,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = (
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + limits.model_max_limits_host
        )
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_03_allocate_single_sparse_system_jacobians_with_contacts(self):
        # Construct the test problem
        model, _data, _state, _limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = (
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + 3 * contacts.model_max_contacts_host
        )
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_04_allocate_single_sparse_system_jacobians_with_limits_and_contacts(self):
        # Construct the test problem
        model, _data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        model_num_cts = (
            model.size.sum_of_num_bilateral_joint_cts
            + model.size.sum_of_num_bounded_joint_cts
            + limits.model_max_limits_host
            + 3 * contacts.model_max_contacts_host
        )
        model_num_dofs = model.size.sum_of_num_joint_dofs
        model_num_bodies = model.size.sum_of_num_bodies
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, 1)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, 1)
        self.assertTrue((jacobians._J_cts.bsm.max_dims.numpy() == [[model_num_cts, 6 * model_num_bodies]]).all())
        self.assertTrue((jacobians._J_dofs.bsm.max_dims.numpy() == [[model_num_dofs, 6 * model_num_bodies]]).all())
        self.assertEqual(jacobians._J_cts.bsm.max_nzb.numpy()[0], 2 * model_num_cts)

    def test_05_allocate_homogeneous_sparse_system_jacobians(self):
        # Construct the test problem
        model, _data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, model.size.num_worlds)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, model.size.num_worlds)
        self.assertTrue(
            (
                jacobians._J_cts.bsm.max_dims.numpy()
                == [[max_total_cts[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )
        self.assertTrue(
            (
                jacobians._J_dofs.bsm.max_dims.numpy()
                == [[num_joint_dofs[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )
        self.assertTrue(
            (jacobians._J_cts.bsm.max_nzb.numpy() == [2 * max_total_cts[w] for w in range(model.size.num_worlds)]).all()
        )

    def test_06_allocate_heterogeneous_sparse_system_jacobians(self):
        # Construct the test problem
        model, _data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        if self.verbose:
            print(f"J_cts max_dims (shape={jacobians._J_cts.bsm.max_dims.shape}): {jacobians._J_cts.bsm.max_dims}")
            print(f"J_cts dims (shape={jacobians._J_cts.bsm.dims.shape}): {jacobians._J_cts.bsm.dims}")
            print(f"J_cts max_nzb (shape={jacobians._J_cts.bsm.max_nzb.shape}): {jacobians._J_cts.bsm.max_nzb}")
            print(f"J_dofs max_dims (shape={jacobians._J_dofs.bsm.max_dims.shape}): {jacobians._J_dofs.bsm.max_dims}")
            print(f"J_dofs dims (shape={jacobians._J_dofs.bsm.dims.shape}): {jacobians._J_dofs.bsm.dims}")
            print(f"J_dofs max_nzb (shape={jacobians._J_dofs.bsm.max_nzb.shape}): {jacobians._J_dofs.bsm.max_nzb}")

        # Check the allocation of Jacobians
        num_body_dofs = model.info.num_body_dofs.numpy().tolist()
        num_joint_dofs = model.info.num_joint_dofs.numpy().tolist()
        max_total_cts = model.info.max_total_cts.numpy().tolist()
        self.assertEqual(jacobians._J_cts.bsm.num_matrices, model.size.num_worlds)
        self.assertEqual(jacobians._J_dofs.bsm.num_matrices, model.size.num_worlds)
        self.assertTrue(
            (
                jacobians._J_cts.bsm.max_dims.numpy()
                == [[max_total_cts[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )
        self.assertTrue(
            (
                jacobians._J_dofs.bsm.max_dims.numpy()
                == [[num_joint_dofs[w], num_body_dofs[w]] for w in range(model.size.num_worlds)]
            ).all()
        )

    def test_07_build_compare_single_system_jacobians(self):
        # Construct the test problem
        model, data, *_ = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model)
        jacobians_dense = DenseSystemJacobians(model=model)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data)
        jacobians_dense.build(model=model, data=data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, None, None, jacobians_dense, jacobians)

    def test_08_build_compare_single_system_jacobians_with_limits(self):
        # Construct the test problem
        model, data, _state, limits, _contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=True,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, None, jacobians_dense, jacobians)

    def test_09_build_compare_single_system_jacobians_with_contacts(self):
        # Construct the test problem
        model, data, _state, _limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, None, contacts, jacobians_dense, jacobians)

    def test_10_build_compare_single_system_jacobians_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, contacts, jacobians_dense, jacobians)

    def test_11_build_compare_homogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, contacts, jacobians_dense, jacobians)

    def test_12_build_compare_heterogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians_sparse = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        jacobians_dense = DenseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians_sparse.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        jacobians_dense.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Check that Jacobians match
        self._compare_dense_sparse_jacobians(model, limits, contacts, jacobians_dense, jacobians_sparse)

    def test_13_build_col_major_single_system_jacobians(self):
        # Construct the test problem
        model, data, *_ = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(model=model, jacobians=jacobians)
        jacobian_col_maj.update(model=model, jacobians=jacobians)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_14_build_col_major_single_system_jacobians_with_limits(self):
        # Construct the test problem
        model, data, _state, limits, _contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=False,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(model=model, limits=limits, jacobians=jacobians)
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_15_build_col_major_single_system_jacobians_with_contacts(self):
        # Construct the test problem
        model, data, _state, _limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=False,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(model=model, contacts=contacts, jacobians=jacobians)
        jacobian_col_maj.update(model=model, jacobians=jacobians, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_16_build_col_major_single_system_jacobians_with_limits_and_contacts(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=1,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians
        )
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_17_build_col_major_homogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            max_world_contacts=12,
            num_worlds=3,
            with_limits=True,
            with_contacts=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians
        )
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)

    def test_18_build_col_major_heterogeneous_system_jacobians(self):
        # Construct the test problem
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            max_world_contacts=12,
            with_limits=True,
            with_contacts=True,
            with_implicit_joints=True,
            verbose=self.verbose,
        )

        # Create the Jacobians container
        jacobians = SparseSystemJacobians(model=model, limits=limits, contacts=contacts)
        wp.synchronize()

        # Build the system Jacobians
        jacobians.build(model=model, data=data, limits=limits.data, contacts=contacts.data)
        wp.synchronize()

        # Build column-major constraint Jacobian version
        jacobian_col_maj = ColMajorSparseConstraintJacobians(
            model=model, limits=limits, contacts=contacts, jacobians=jacobians
        )
        jacobian_col_maj.update(model=model, jacobians=jacobians, limits=limits, contacts=contacts)

        # Check that Jacobians match
        self._compare_row_col_major_jacobians(jacobians, jacobian_col_maj)


class TestKinematicsAnalyticSystemJacobians(unittest.TestCase):
    """Verify analytic joint Jacobians match residual time derivatives."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _build_joint_jacobians(
        self,
        model,
        data,
        limits,
        contacts,
        sparse: bool,
        method: ConstraintJacobianMethod,
    ):
        """Build Jacobians for the given method and storage format."""
        if sparse:
            jacobians = SparseSystemJacobians(
                model=model,
                limits=limits,
                contacts=contacts,
                constraint_jacobian_method=method,
            )
        else:
            jacobians = DenseSystemJacobians(
                model=model,
                limits=limits,
                contacts=contacts,
                constraint_jacobian_method=method,
            )
        jacobians.build(
            model=model,
            data=data,
            limits=limits.data if limits is not None else None,
            contacts=contacts.data if contacts is not None else None,
        )
        wp.synchronize()
        return jacobians

    def _integrate_body_transforms(self, transforms: np.ndarray, twists: np.ndarray, dt: float) -> np.ndarray:
        """Apply world-frame body twists using the production log-map pose update."""
        device = self.default_device
        poses_in = wp.array(transforms, dtype=wp.transformf, device=device)
        twists_in = wp.array(twists, dtype=wp.spatial_vectorf, device=device)
        poses_out = wp.empty_like(poses_in)
        wp.launch(
            _integrate_body_poses_logmap,
            dim=transforms.shape[0],
            inputs=[poses_in, twists_in, wp.float32(dt), poses_out],
            device=device,
        )
        return poses_out.numpy()

    def _finite_difference_joint_constraint_rates(
        self,
        model,
        data,
        dt: float = 1e-4,
    ) -> np.ndarray:
        """Estimate ``dr_j/dt`` from a log-map pose step along ``data.bodies.u_i``."""
        r0 = data.joints.r_j.numpy().copy()
        q0 = data.bodies.q_i.numpy().copy()
        u = data.bodies.u_i.numpy().copy()
        q1 = self._integrate_body_transforms(q0, u, dt)

        data_pert = model.data()
        data_pert.bodies.q_i.assign(q1)
        compute_joints_data(model=model, data=data_pert, q_j_p=wp.zeros_like(data.joints.q_j))
        wp.synchronize()
        r1 = data_pert.joints.r_j.numpy()
        return (r1 - r0) / dt

    def _set_nonzero_body_twists(self, model, data):
        """Assign non-zero twists so the chain-rule check is non-trivial."""
        u_host = data.bodies.u_i.numpy().copy()
        bodies_offset = model.info.bodies_offset.numpy()
        num_body_dofs = model.info.num_body_dofs.numpy()
        for wid in range(model.info.num_worlds):
            bio = int(bodies_offset[wid])
            num_bodies = int(num_body_dofs[wid]) // 6
            for loc in range(num_bodies):
                bid = bio + loc
                u_host[bid] = np.array(
                    [0.1, -0.05, 0.02, 0.05 * loc, 0.4 + 0.03 * loc, 0.15 - 0.02 * loc],
                    dtype=np.float32,
                )
        data.bodies.u_i.assign(u_host)
        compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
        wp.synchronize()

    def _prepare_chain_rule_state(self, model, data):
        """Assign non-zero twists and a small pose tilt for finite rotation residuals."""
        self._set_nonzero_body_twists(model, data)
        q_host = data.bodies.q_i.numpy().copy()
        bodies_offset = model.info.bodies_offset.numpy()
        num_body_dofs = model.info.num_body_dofs.numpy()
        for wid in range(model.info.num_worlds):
            bio = int(bodies_offset[wid])
            num_bodies = int(num_body_dofs[wid]) // 6
            for loc in range(num_bodies):
                bid = bio + loc
                q_old = wp.quatf(q_host[bid, 3], q_host[bid, 4], q_host[bid, 5], q_host[bid, 6])
                tilt = quat_exp(wp.vec3f(0.02 * loc, 0.12 + 0.03 * loc, 0.05 * loc))
                q_new = tilt * q_old
                q_host[bid, 3] = q_new.x
                q_host[bid, 4] = q_new.y
                q_host[bid, 5] = q_new.z
                q_host[bid, 6] = q_new.w
        data.bodies.q_i.assign(q_host)
        compute_joints_data(model=model, data=data, q_j_p=wp.zeros_like(data.joints.q_j))
        wp.synchronize()

    def _assert_analytic_satisfies_residual_chain_rule_with_tilt(self, model, data, limits, contacts, sparse: bool):
        """Like ``_assert_analytic_satisfies_residual_chain_rule`` but with pose tilt."""
        self._prepare_chain_rule_state(model, data)
        jacobian = self._build_joint_jacobians(model, data, limits, contacts, sparse, ConstraintJacobianMethod.ANALYTIC)
        dr_numeric = self._finite_difference_joint_constraint_rates(model, data)
        dr_analytic = self._kinematic_constraint_rates_from_jacobian(model, data, jacobian)
        np.testing.assert_allclose(dr_analytic, dr_numeric, rtol=5e-2, atol=5e-2)

    def _assert_analytic_satisfies_residual_chain_rule_for_dof_types(
        self,
        model,
        data,
        limits,
        contacts,
        sparse: bool,
        dof_types: set[int],
        *,
        use_tilt: bool = False,
    ):
        """Verify ``J_cts @ u`` for kinematic rows belonging to selected joint DoF types."""
        if use_tilt:
            self._prepare_chain_rule_state(model, data)
        else:
            self._set_nonzero_body_twists(model, data)
        jacobian = self._build_joint_jacobians(model, data, limits, contacts, sparse, ConstraintJacobianMethod.ANALYTIC)
        dr_numeric = self._finite_difference_joint_constraint_rates(model, data)
        dr_analytic = self._kinematic_constraint_rates_from_jacobian(model, data, jacobian)
        dof_type = model.joints.dof_type.numpy()
        num_kin = model.joints.num_kinematic_cts.numpy()
        ana_sel = []
        num_sel = []
        idx = 0
        for jid in range(model.size.sum_of_num_joints):
            dof = int(dof_type[jid])
            for local in range(int(num_kin[jid])):
                if dof in dof_types:
                    ana_sel.append(dr_analytic[idx])
                    num_sel.append(dr_numeric[idx])
                idx += 1
        np.testing.assert_allclose(np.array(ana_sel), np.array(num_sel), rtol=5e-2, atol=5e-2)

    def _kinematic_constraint_rates_from_jacobian(self, model, data, jacobian) -> np.ndarray:
        """Compute ``dr_j/dt`` from ``J_cts`` for kinematic joint constraints only."""
        J_cts = extract_cts_jacobians(model, None, None, jacobian)
        u_host = data.bodies.u_i.numpy()
        bodies_offset = model.info.bodies_offset.numpy()
        num_body_dofs = model.info.num_body_dofs.numpy()
        world_cts_offset = model.info.total_cts_offset.numpy()
        joint_wid = model.joints.wid.numpy()
        kin_total = model.joints.kinematic_cts_offset_total_cts.numpy()
        num_kin = model.joints.num_kinematic_cts.numpy()

        rates = []
        for jid in range(model.size.sum_of_num_joints):
            wid = joint_wid[jid]
            bio = bodies_offset[wid]
            num_bodies = num_body_dofs[wid] // 6
            u_w = u_host[bio : bio + num_bodies].reshape(-1)
            row_base = kin_total[jid] - world_cts_offset[wid]
            for i in range(num_kin[jid]):
                rates.append(J_cts[wid][row_base + i] @ u_w)
        return np.array(rates, dtype=np.float32)

    def _assert_analytic_satisfies_residual_chain_rule(self, model, data, limits, contacts, sparse: bool):
        """Verify ``J_cts @ u`` matches the time derivative of joint constraint residuals."""
        self._set_nonzero_body_twists(model, data)
        jacobian = self._build_joint_jacobians(model, data, limits, contacts, sparse, ConstraintJacobianMethod.ANALYTIC)
        dr_numeric = self._finite_difference_joint_constraint_rates(model, data)
        dr_analytic = self._kinematic_constraint_rates_from_jacobian(model, data, jacobian)
        np.testing.assert_allclose(dr_analytic, dr_numeric, rtol=5e-2, atol=5e-2)

    def test_analytic_residual_chain_rule_dense_fourbar(self):
        """Verify analytic dense ``J_cts`` matches ``dr_j/dt`` from pose perturbation on fourbar."""
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_satisfies_residual_chain_rule(model, data, limits, contacts, sparse=False)

    def test_analytic_residual_chain_rule_sparse_fourbar(self):
        """Verify analytic sparse ``J_cts`` matches ``dr_j/dt`` from pose perturbation on fourbar."""
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_satisfies_residual_chain_rule(model, data, limits, contacts, sparse=True)

    def test_analytic_residual_chain_rule_dense_heterogeneous(self):
        """Verify analytic dense ``J_cts`` matches ``dr_j/dt`` on a heterogeneous model."""
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            with_limits=False,
            with_contacts=False,
            with_implicit_joints=True,
        )
        self._assert_analytic_satisfies_residual_chain_rule(model, data, limits, contacts, sparse=False)

    def test_analytic_residual_chain_rule_binary_universal(self):
        """Verify analytic ``J_cts`` on a binary universal joint with FK-style angular rows."""
        builder = build_binary_universal_joint_test(limits=False, ground=False)
        model, data, _state, limits, contacts = make_test_problem(
            builder=builder,
            device=self.default_device,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_satisfies_residual_chain_rule_for_dof_types(
            model,
            data,
            limits,
            contacts,
            sparse=False,
            dof_types={int(JointDoFType.UNIVERSAL)},
            use_tilt=True,
        )

    def test_analytic_residual_chain_rule_passive_universal(self):
        """Verify analytic ``J_cts`` on a passive universal joint with FK orthogonality rows."""
        builder = build_binary_universal_joint_test(
            limits=False, ground=False, act_type=JointActuationType.PASSIVE
        )
        model, data, _state, limits, contacts = make_test_problem(
            builder=builder,
            device=self.default_device,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_satisfies_residual_chain_rule_for_dof_types(
            model,
            data,
            limits,
            contacts,
            sparse=False,
            dof_types={int(JointDoFType.UNIVERSAL)},
            use_tilt=True,
        )

    def test_analytic_residual_chain_rule_fixed_joint(self):
        """Verify analytic ``J_cts`` on fixed-joint ``quat_log`` rows with FK frame transform."""
        builder = build_binary_revolute_joint_test(limits=False, ground=False, implicit_pd=False)
        model, data, _state, limits, contacts = make_test_problem(
            builder=builder,
            device=self.default_device,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_satisfies_residual_chain_rule_for_dof_types(
            model,
            data,
            limits,
            contacts,
            sparse=False,
            dof_types={int(JointDoFType.FIXED)},
            use_tilt=True,
        )

    def test_analytic_differs_from_geometric_on_rotational_rows(self):
        """Verify residual-based rotation rows differ from the screw assembly at finite pose."""
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
        )
        jacobian_geo = self._build_joint_jacobians(
            model, data, limits, contacts, sparse=False, method=ConstraintJacobianMethod.GEOMETRIC
        )
        jacobian_ana = self._build_joint_jacobians(
            model, data, limits, contacts, sparse=False, method=ConstraintJacobianMethod.ANALYTIC
        )
        J_cts_geo = extract_cts_jacobians(model, None, None, jacobian_geo)[0]
        J_cts_ana = extract_cts_jacobians(model, None, None, jacobian_ana)[0]
        self.assertFalse(np.allclose(J_cts_geo, J_cts_ana, rtol=1e-5, atol=1e-5))

    def _assert_analytic_actuation_matches_geometric(self, model, data, limits, contacts, sparse: bool):
        """Verify analytic ``J_dofs`` matches geometric screw assembly."""
        jacobian_geo = self._build_joint_jacobians(
            model, data, limits, contacts, sparse, ConstraintJacobianMethod.GEOMETRIC
        )
        jacobian_ana = self._build_joint_jacobians(
            model, data, limits, contacts, sparse, ConstraintJacobianMethod.ANALYTIC
        )
        J_dofs_geo = extract_dofs_jacobians(model, jacobians=jacobian_geo)
        J_dofs_ana = extract_dofs_jacobians(model, jacobians=jacobian_ana)
        for wid in range(model.info.num_worlds):
            np.testing.assert_allclose(J_dofs_ana[wid], J_dofs_geo[wid], rtol=1e-5, atol=1e-5)

    def test_analytic_actuation_matches_geometric_dense_fourbar(self):
        """Verify analytic dense ``J_dofs`` matches geometric screw assembly on fourbar."""
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_actuation_matches_geometric(model, data, limits, contacts, sparse=False)

    def test_analytic_actuation_matches_geometric_sparse_fourbar(self):
        """Verify analytic sparse ``J_dofs`` matches geometric screw assembly on fourbar."""
        model, data, _state, limits, contacts = make_test_problem_fourbar(
            device=self.default_device,
            num_worlds=1,
            with_limits=False,
            with_contacts=False,
        )
        self._assert_analytic_actuation_matches_geometric(model, data, limits, contacts, sparse=True)

    def test_analytic_actuation_matches_geometric_dense_heterogeneous(self):
        """Verify analytic dense ``J_dofs`` matches geometric screw assembly on heterogeneous model."""
        model, data, _state, limits, contacts = make_test_problem_heterogeneous(
            device=self.default_device,
            with_limits=False,
            with_contacts=False,
            with_implicit_joints=True,
        )
        self._assert_analytic_actuation_matches_geometric(model, data, limits, contacts, sparse=False)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
