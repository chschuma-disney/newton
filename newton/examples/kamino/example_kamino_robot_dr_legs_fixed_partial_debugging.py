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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import warp as wp

import newton
import newton.examples


def _build_index_remap(kept_indices: Sequence[int]) -> dict[int, int]:
    return {old: new for new, old in enumerate(kept_indices)}


def _resolve_body_indices(builder: newton.ModelBuilder, body_labels: Sequence[str]) -> set[int]:
    """Resolve body labels (full prim paths or short names) to body indices."""
    label_to_index = {label: index for index, label in enumerate(builder.body_label)}
    short_name_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(builder.body_label):
        short_name_to_indices.setdefault(label.split("/")[-1], []).append(index)

    remove_indices: set[int] = set()
    for label in body_labels:
        if label in label_to_index:
            remove_indices.add(label_to_index[label])
            continue

        short_name = label.split("/")[-1]
        if short_name in short_name_to_indices:
            remove_indices.update(short_name_to_indices[short_name])
            continue

        for index, body_label in enumerate(builder.body_label):
            if label in body_label:
                remove_indices.add(index)

    return remove_indices


def _joint_slice_bounds(starts: list[int], total_count: int, joint_id: int, joint_count: int) -> tuple[int, int]:
    start = starts[joint_id]
    if joint_id < joint_count - 1:
        return start, starts[joint_id + 1]
    return start, total_count


def _compact_parallel_lists(kept_indices: Sequence[int], lists: Sequence[list[Any]]) -> None:
    for values in lists:
        compacted = [values[index] for index in kept_indices]
        values.clear()
        values.extend(compacted)


def _remap_dict_keys(mapping: dict[int, int], values: dict[int, Any] | None) -> dict[int, Any]:
    if not values:
        return {}
    return {mapping[key]: value for key, value in values.items() if key in mapping}


def _remap_reference_value(value: Any, mapping: dict[int, int]) -> Any:
    if value is None:
        return value
    try:
        index = int(value)
    except (TypeError, ValueError):
        return value
    if index == -1:
        return value
    return mapping.get(index, -1)


def remove_bodies_from_builder(builder: newton.ModelBuilder, body_labels: Sequence[str]) -> None:
    """Remove rigid bodies and all dependent builder state.

    Joints incident on a removed body are dropped. Shapes parented to removed bodies are
    dropped. Equality and mimic constraints that reference a removed body or joint are
    removed. Remaining entity indices are compacted and remapped across builder arrays and
    custom attributes.
    """
    remove_body_ids = _resolve_body_indices(builder, body_labels)
    if not remove_body_ids:
        return

    bodies_to_keep = [body_id for body_id in range(builder.body_count) if body_id not in remove_body_ids]
    body_remap = _build_index_remap(bodies_to_keep)

    joints_to_keep = [
        joint_id
        for joint_id in range(builder.joint_count)
        if builder.joint_parent[joint_id] not in remove_body_ids
        and builder.joint_child[joint_id] not in remove_body_ids
    ]
    joint_remap = _build_index_remap(joints_to_keep)

    old_shape_body = list(builder.shape_body)
    shapes_to_keep = [shape_id for shape_id, body_id in enumerate(old_shape_body) if body_id not in remove_body_ids]
    shape_remap = _build_index_remap(shapes_to_keep)

    dof_remap: dict[int, int] = {}
    coord_remap: dict[int, int] = {}
    cts_remap: dict[int, int] = {}

    original_joint_articulation = list(builder.joint_articulation) if builder.joint_articulation else []
    original_joint_world = list(builder.joint_world) if builder.joint_world else []
    original_body_world = list(builder.body_world) if builder.body_world else []
    original_body_color_groups = list(builder.body_color_groups) if builder.body_color_groups else []

    original_joint_q = list(builder.joint_q)
    original_joint_target_q = list(builder.joint_target_q)
    original_joint_qd = list(builder.joint_qd)
    original_joint_target_qd = list(builder.joint_target_qd)
    original_joint_cts = list(builder.joint_cts)
    original_joint_armature = list(builder.joint_armature)
    original_joint_axis = list(builder.joint_axis)
    original_joint_target_mode = list(builder.joint_target_mode)
    original_joint_target_ke = list(builder.joint_target_ke)
    original_joint_target_kd = list(builder.joint_target_kd)
    original_joint_damping = list(builder.joint_damping)
    original_joint_limit_lower = list(builder.joint_limit_lower)
    original_joint_limit_upper = list(builder.joint_limit_upper)
    original_joint_limit_ke = list(builder.joint_limit_ke)
    original_joint_limit_kd = list(builder.joint_limit_kd)
    original_joint_effort_limit = list(builder.joint_effort_limit)
    original_joint_velocity_limit = list(builder.joint_velocity_limit)
    original_joint_friction = list(builder.joint_friction)

    new_joint_label: list[str] = []
    new_joint_type: list[int] = []
    new_joint_parent: list[int] = []
    new_joint_child: list[int] = []
    new_joint_q: list[float] = []
    new_joint_target_q: list[float] = []
    new_joint_qd: list[float] = []
    new_joint_target_qd: list[float] = []
    new_joint_cts: list[float] = []
    new_joint_q_start: list[int] = []
    new_joint_qd_start: list[int] = []
    new_joint_cts_start: list[int] = []
    new_joint_enabled: list[bool] = []
    new_joint_collision_filter_parent: list[bool] = []
    new_joint_armature: list[float] = []
    new_joint_X_p: list[Any] = []
    new_joint_X_c: list[Any] = []
    new_joint_axis: list[Any] = []
    new_joint_target_mode: list[int] = []
    new_joint_target_ke: list[float] = []
    new_joint_target_kd: list[float] = []
    new_joint_damping: list[float] = []
    new_joint_limit_lower: list[float] = []
    new_joint_limit_upper: list[float] = []
    new_joint_limit_ke: list[float] = []
    new_joint_limit_kd: list[float] = []
    new_joint_effort_limit: list[float] = []
    new_joint_dof_dim: list[tuple[int, int]] = []
    new_joint_world: list[int] = []
    new_joint_articulation: list[int] = []

    for joint_id in joints_to_keep:
        q_start, q_end = _joint_slice_bounds(
            builder.joint_q_start, len(original_joint_q), joint_id, builder.joint_count
        )
        qd_start, qd_end = _joint_slice_bounds(
            builder.joint_qd_start, len(original_joint_qd), joint_id, builder.joint_count
        )
        cts_start, cts_end = _joint_slice_bounds(
            builder.joint_cts_start, len(original_joint_cts), joint_id, builder.joint_count
        )

        new_q_start = len(new_joint_q)
        new_qd_start = len(new_joint_qd)
        new_cts_start = len(new_joint_cts)

        for old_index in range(q_start, q_end):
            coord_remap[old_index] = len(new_joint_q) + (old_index - q_start)
        for old_index in range(qd_start, qd_end):
            dof_remap[old_index] = len(new_joint_qd) + (old_index - qd_start)
        for old_index in range(cts_start, cts_end):
            cts_remap[old_index] = len(new_joint_cts) + (old_index - cts_start)

        new_joint_label.append(builder.joint_label[joint_id])
        new_joint_type.append(builder.joint_type[joint_id])
        new_joint_parent.append(
            -1 if builder.joint_parent[joint_id] == -1 else body_remap[builder.joint_parent[joint_id]]
        )
        new_joint_child.append(body_remap[builder.joint_child[joint_id]])
        new_joint_q_start.append(new_q_start)
        new_joint_qd_start.append(new_qd_start)
        new_joint_cts_start.append(new_cts_start)
        new_joint_q.extend(original_joint_q[q_start:q_end])
        new_joint_target_q.extend(original_joint_target_q[q_start:q_end])
        new_joint_qd.extend(original_joint_qd[qd_start:qd_end])
        new_joint_target_qd.extend(original_joint_target_qd[qd_start:qd_end])
        new_joint_cts.extend(original_joint_cts[cts_start:cts_end])
        new_joint_armature.extend(original_joint_armature[qd_start:qd_end])
        new_joint_enabled.append(builder.joint_enabled[joint_id])
        new_joint_collision_filter_parent.append(builder.joint_collision_filter_parent[joint_id])
        new_joint_X_p.append(builder.joint_X_p[joint_id])
        new_joint_X_c.append(builder.joint_X_c[joint_id])
        new_joint_dof_dim.append(builder.joint_dof_dim[joint_id])
        if original_joint_world:
            new_joint_world.append(original_joint_world[joint_id])
        if original_joint_articulation:
            new_joint_articulation.append(original_joint_articulation[joint_id])

        num_lin_axes, num_ang_axes = builder.joint_dof_dim[joint_id]
        for axis_index in range(qd_start, qd_start + num_lin_axes + num_ang_axes):
            new_joint_axis.append(original_joint_axis[axis_index])
            new_joint_target_mode.append(original_joint_target_mode[axis_index])
            new_joint_target_ke.append(original_joint_target_ke[axis_index])
            new_joint_target_kd.append(original_joint_target_kd[axis_index])
            new_joint_damping.append(original_joint_damping[axis_index])
            new_joint_limit_lower.append(original_joint_limit_lower[axis_index])
            new_joint_limit_upper.append(original_joint_limit_upper[axis_index])
            new_joint_limit_ke.append(original_joint_limit_ke[axis_index])
            new_joint_limit_kd.append(original_joint_limit_kd[axis_index])
            new_joint_effort_limit.append(original_joint_effort_limit[axis_index])

    builder.joint_label = new_joint_label
    builder.joint_type = new_joint_type
    builder.joint_parent = new_joint_parent
    builder.joint_child = new_joint_child
    builder.joint_q = new_joint_q
    builder.joint_target_q = new_joint_target_q
    builder.joint_qd = new_joint_qd
    builder.joint_target_qd = new_joint_target_qd
    builder.joint_cts = new_joint_cts
    builder.joint_q_start = new_joint_q_start
    builder.joint_qd_start = new_joint_qd_start
    builder.joint_cts_start = new_joint_cts_start
    builder.joint_enabled = new_joint_enabled
    builder.joint_collision_filter_parent = new_joint_collision_filter_parent
    builder.joint_armature = new_joint_armature
    builder.joint_X_p = new_joint_X_p
    builder.joint_X_c = new_joint_X_c
    builder.joint_axis = new_joint_axis
    builder.joint_target_mode = new_joint_target_mode
    builder.joint_target_ke = new_joint_target_ke
    builder.joint_target_kd = new_joint_target_kd
    builder.joint_damping = new_joint_damping
    builder.joint_limit_lower = new_joint_limit_lower
    builder.joint_limit_upper = new_joint_limit_upper
    builder.joint_limit_ke = new_joint_limit_ke
    builder.joint_limit_kd = new_joint_limit_kd
    builder.joint_effort_limit = new_joint_effort_limit
    builder.joint_dof_dim = new_joint_dof_dim
    builder.joint_world = new_joint_world
    builder.joint_articulation = new_joint_articulation
    builder.joint_dof_count = len(builder.joint_qd)
    builder.joint_coord_count = len(builder.joint_q)
    builder.joint_constraint_count = len(builder.joint_cts)

    if original_joint_velocity_limit:
        builder.joint_velocity_limit = [original_joint_velocity_limit[index] for index in sorted(dof_remap)]
    if original_joint_friction:
        builder.joint_friction = [original_joint_friction[index] for index in sorted(dof_remap)]
    if builder.joint_f:
        builder.joint_f = [builder.joint_f[index] for index in sorted(dof_remap)] if dof_remap else []
    if builder.joint_act:
        builder.joint_act = [builder.joint_act[index] for index in sorted(dof_remap)] if dof_remap else []

    _compact_parallel_lists(
        bodies_to_keep,
        [
            builder.body_mass,
            builder.body_inertia,
            builder.body_inv_mass,
            builder.body_inv_inertia,
            builder.body_com,
            builder.body_q,
            builder.body_qd,
            builder.body_label,
            builder.body_lock_inertia,
            builder.body_flags,
        ],
    )
    if original_body_world:
        builder.body_world = [original_body_world[body_id] for body_id in bodies_to_keep]

    static_shapes = builder.body_shapes.get(-1, [])
    builder.body_shapes.clear()
    builder.body_shapes[-1] = list(static_shapes)
    for new_body_id, _old_body_id in enumerate(bodies_to_keep):
        builder.body_shapes[new_body_id] = []

    shape_lists = [
        builder.shape_label,
        builder.shape_transform,
        builder.shape_flags,
        builder.shape_type,
        builder.shape_scale,
        builder.shape_source,
        builder.shape_color,
        builder.shape_is_solid,
        builder.shape_margin,
        builder.shape_material_ke,
        builder.shape_material_kd,
        builder.shape_material_kf,
        builder.shape_material_ka,
        builder.shape_material_mu,
        builder.shape_material_restitution,
        builder.shape_material_mu_torsional,
        builder.shape_material_mu_rolling,
        builder.shape_material_kh,
        builder.shape_gap,
        builder.shape_collision_group,
        builder.shape_collision_radius,
        builder.shape_world,
        builder.shape_sdf_narrow_band_range,
        builder.shape_sdf_target_voxel_size,
        builder.shape_sdf_max_resolution,
        builder.shape_sdf_texture_format,
        builder.shape_sdf_padding,
    ]
    _compact_parallel_lists(shapes_to_keep, shape_lists)
    builder.shape_body = [
        -1 if old_shape_body[shape_id] == -1 else body_remap[old_shape_body[shape_id]] for shape_id in shapes_to_keep
    ]

    for new_shape_id, _old_shape_id in enumerate(shapes_to_keep):
        body_id = builder.shape_body[new_shape_id]
        if body_id >= 0:
            builder.body_shapes[body_id].append(new_shape_id)

    if builder.shape_collision_filter_pairs:
        builder.shape_collision_filter_pairs = [
            (shape_remap[shape_a], shape_remap[shape_b])
            for shape_a, shape_b in builder.shape_collision_filter_pairs
            if shape_a in shape_remap and shape_b in shape_remap
        ]

    if original_body_color_groups:
        remapped_groups: list[list[int]] = []
        for group in original_body_color_groups:
            remapped_group = [body_remap[body_id] for body_id in group if body_id in body_remap]
            if remapped_group:
                remapped_groups.append(remapped_group)
        builder.body_color_groups = remapped_groups

    articulation_first_joint: dict[int, int] = {}
    articulation_last_joint: dict[int, int] = {}
    for new_joint_id, old_joint_id in enumerate(joints_to_keep):
        if not original_joint_articulation or old_joint_id >= len(original_joint_articulation):
            continue
        old_articulation = original_joint_articulation[old_joint_id]
        if old_articulation < 0:
            continue
        articulation_first_joint.setdefault(old_articulation, new_joint_id)
        articulation_last_joint[old_articulation] = new_joint_id

    original_articulation_start = list(builder.articulation_start)
    original_articulation_label = list(builder.articulation_label)
    original_articulation_world = list(builder.articulation_world)
    articulation_remap: dict[int, int] = {}
    builder.articulation_start = []
    builder.articulation_end = []
    builder.articulation_label = []
    builder.articulation_world = []
    for articulation_id in range(len(original_articulation_start)):
        if articulation_id not in articulation_first_joint:
            continue
        articulation_remap[articulation_id] = len(builder.articulation_start)
        builder.articulation_start.append(articulation_first_joint[articulation_id])
        builder.articulation_end.append(articulation_last_joint[articulation_id] + 1)
        if articulation_id < len(original_articulation_label):
            builder.articulation_label.append(original_articulation_label[articulation_id])
        if articulation_id < len(original_articulation_world):
            builder.articulation_world.append(original_articulation_world[articulation_id])
    builder.joint_articulation = [
        articulation_remap.get(articulation_id, -1) if articulation_id >= 0 else -1
        for articulation_id in builder.joint_articulation
    ]

    for cable_index in range(len(builder._cable_label)):
        if builder._cable_body_end[cable_index] > builder._cable_body_start[cable_index]:
            cable_bodies = range(builder._cable_body_start[cable_index], builder._cable_body_end[cable_index])
            if any(body_id in remove_body_ids for body_id in cable_bodies):
                raise ValueError(
                    f"Cable group {builder._cable_label[cable_index]!r} references a removed body; "
                    "remove the cable group explicitly before removing bodies."
                )
            builder._cable_body_start[cable_index] = body_remap[builder._cable_body_start[cable_index]]
            builder._cable_body_end[cable_index] = body_remap[builder._cable_body_end[cable_index] - 1] + 1
        if builder._cable_joint_end[cable_index] > builder._cable_joint_start[cable_index]:
            first_joint = builder._cable_joint_start[cable_index]
            last_joint = builder._cable_joint_end[cable_index] - 1
            if first_joint not in joint_remap or last_joint not in joint_remap:
                raise ValueError(
                    f"Cable group {builder._cable_label[cable_index]!r} references a removed joint; "
                    "remove the cable group explicitly before removing bodies."
                )
            builder._cable_joint_start[cable_index] = joint_remap[first_joint]
            builder._cable_joint_end[cable_index] = joint_remap[last_joint] + 1
        else:
            boundary = builder._cable_joint_start[cable_index]
            builder._cable_joint_start[cable_index] = sum(1 for old_joint in joint_remap if old_joint < boundary)
            builder._cable_joint_end[cable_index] = builder._cable_joint_start[cable_index]

    mimics_to_keep = [
        mimic_id
        for mimic_id in range(len(builder.constraint_mimic_joint0))
        if (builder.constraint_mimic_joint0[mimic_id] == -1 or builder.constraint_mimic_joint0[mimic_id] in joint_remap)
        and (
            builder.constraint_mimic_joint1[mimic_id] == -1 or builder.constraint_mimic_joint1[mimic_id] in joint_remap
        )
    ]
    mimic_remap = _build_index_remap(mimics_to_keep)
    _compact_parallel_lists(
        mimics_to_keep,
        [
            builder.constraint_mimic_joint0,
            builder.constraint_mimic_joint1,
            builder.constraint_mimic_coef0,
            builder.constraint_mimic_coef1,
            builder.constraint_mimic_enabled,
            builder.constraint_mimic_label,
            builder.constraint_mimic_world,
        ],
    )
    builder.constraint_mimic_joint0 = [
        joint_remap[joint_id] if joint_id != -1 else -1 for joint_id in builder.constraint_mimic_joint0
    ]
    builder.constraint_mimic_joint1 = [
        joint_remap[joint_id] if joint_id != -1 else -1 for joint_id in builder.constraint_mimic_joint1
    ]

    muscles_to_keep: list[int] = []
    for muscle_id, start in enumerate(builder.muscle_start):
        end = (
            builder.muscle_start[muscle_id + 1]
            if muscle_id + 1 < len(builder.muscle_start)
            else len(builder.muscle_bodies)
        )
        waypoint_bodies = builder.muscle_bodies[start:end]
        if any(body_id in remove_body_ids for body_id in waypoint_bodies):
            continue
        muscles_to_keep.append(muscle_id)
    if muscles_to_keep:
        new_muscle_start: list[int] = []
        new_muscle_params: list[tuple[float, float, float, float, float]] = []
        new_muscle_activations: list[float] = []
        new_muscle_bodies: list[int] = []
        new_muscle_points: list[Any] = []
        for _new_muscle_id, old_muscle_id in enumerate(muscles_to_keep):
            old_start = builder.muscle_start[old_muscle_id]
            old_end = (
                builder.muscle_start[old_muscle_id + 1]
                if old_muscle_id + 1 < len(builder.muscle_start)
                else len(builder.muscle_bodies)
            )
            new_muscle_start.append(len(new_muscle_bodies))
            new_muscle_params.append(builder.muscle_params[old_muscle_id])
            new_muscle_activations.append(builder.muscle_activations[old_muscle_id])
            new_muscle_bodies.extend(body_remap[body_id] for body_id in builder.muscle_bodies[old_start:old_end])
            new_muscle_points.extend(builder.muscle_points[old_start:old_end])
        new_muscle_start.append(len(new_muscle_bodies))
        builder.muscle_start = new_muscle_start
        builder.muscle_params = new_muscle_params
        builder.muscle_activations = new_muscle_activations
        builder.muscle_bodies = new_muscle_bodies
        builder.muscle_points = new_muscle_points

    equality_count = builder._equality_constraint_count
    if equality_count > 0:
        body1_values = builder._eq_list("equality_constraint_body1")
        body2_values = builder._eq_list("equality_constraint_body2")
        joint1_values = builder._eq_list("equality_constraint_joint1")
        joint2_values = builder._eq_list("equality_constraint_joint2")
        equality_rows_to_keep = []
        for row in range(equality_count):
            body1 = body1_values[row]
            body2 = body2_values[row]
            joint1 = joint1_values[row]
            joint2 = joint2_values[row]
            if body1 in remove_body_ids or body2 in remove_body_ids:
                continue
            if joint1 != -1 and joint1 not in joint_remap:
                continue
            if joint2 != -1 and joint2 not in joint_remap:
                continue
            equality_rows_to_keep.append(row)

        equality_attrs = [
            attr for attr in builder.custom_attributes.values() if attr.frequency == "mujoco:equality_constraint"
        ]
        for attr in equality_attrs:
            if not attr.values:
                attr.values = []
                continue
            attr.values = [attr.values[row] for row in equality_rows_to_keep if row < len(attr.values)]
        builder._custom_frequency_counts["mujoco:equality_constraint"] = len(equality_rows_to_keep)

        for attr in equality_attrs:
            if attr.references == "body" and isinstance(attr.values, list):
                attr.values = [_remap_reference_value(value, body_remap) for value in attr.values]
            elif attr.references == "joint" and isinstance(attr.values, list):
                attr.values = [_remap_reference_value(value, joint_remap) for value in attr.values]

    frequency_remaps = {
        newton.Model.AttributeFrequency.BODY: body_remap,
        newton.Model.AttributeFrequency.SHAPE: shape_remap,
        newton.Model.AttributeFrequency.JOINT: joint_remap,
        newton.Model.AttributeFrequency.JOINT_DOF: dof_remap,
        newton.Model.AttributeFrequency.JOINT_COORD: coord_remap,
        newton.Model.AttributeFrequency.JOINT_CONSTRAINT: cts_remap,
        newton.Model.AttributeFrequency.ARTICULATION: articulation_remap,
        newton.Model.AttributeFrequency.CONSTRAINT_MIMIC: mimic_remap,
    }
    for custom_attr in builder.custom_attributes.values():
        if custom_attr.frequency in frequency_remaps and isinstance(custom_attr.values, dict):
            custom_attr.values = _remap_dict_keys(frequency_remaps[custom_attr.frequency], custom_attr.values)
        elif custom_attr.references == "body" and isinstance(custom_attr.values, list):
            custom_attr.values = [_remap_reference_value(value, body_remap) for value in custom_attr.values]
        elif custom_attr.references == "joint" and isinstance(custom_attr.values, list):
            custom_attr.values = [_remap_reference_value(value, joint_remap) for value in custom_attr.values]
        elif custom_attr.references == "shape" and isinstance(custom_attr.values, list):
            custom_attr.values = [_remap_reference_value(value, shape_remap) for value in custom_attr.values]
        elif custom_attr.references == "articulation" and isinstance(custom_attr.values, list):
            custom_attr.values = [_remap_reference_value(value, articulation_remap) for value in custom_attr.values]
        elif custom_attr.references == "constraint_mimic" and isinstance(custom_attr.values, list):
            custom_attr.values = [_remap_reference_value(value, mimic_remap) for value in custom_attr.values]

    for entry in builder.actuator_entries.values():
        kept_indices = [index for index, old_dof in enumerate(entry.indices) if old_dof in dof_remap]
        entry.indices = [dof_remap[entry.indices[index]] for index in kept_indices]
        entry.pos_indices = [coord_remap[entry.pos_indices[index]] for index in kept_indices]
        entry.controller_args = [entry.controller_args[index] for index in kept_indices]
        entry.delay_args = [entry.delay_args[index] for index in kept_indices]
        entry.clamping_args = [entry.clamping_args[index] for index in kept_indices]

    builder.joint_parents.clear()
    builder.joint_children.clear()
    for joint_id, (parent, child) in enumerate(zip(builder.joint_parent, builder.joint_child, strict=True)):
        builder.joint_parents.setdefault(child, []).append((parent, joint_id))
        builder.joint_children.setdefault(parent, []).append((child, joint_id))


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

    bodies_to_remove = [
        label
        for label in robot_builder.body_label
        if (short_name := label.split("/")[-1]).endswith("_l") or "_l_" in short_name
    ]
    remove_bodies_from_builder(robot_builder, bodies_to_remove)

    body_labels = [label.split("/")[-1] for label in robot_builder.body_label]

    # foot_l_id = body_labels.index("foot_l")
    foot_r_id = body_labels.index("foot_r")
    # pelvis_id = body_labels.index("pelvis")

    # robot_builder.add_joint_fixed(
    #     -1,
    #     foot_l_id,
    #     parent_xform=robot_builder.body_q[foot_l_id],
    #     child_xform=wp.transform_identity(),
    # )
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
