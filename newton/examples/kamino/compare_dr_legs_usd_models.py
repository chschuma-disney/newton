# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the old and new DR Legs USD model variants in this folder.

Loads both ``model_old`` and ``model_new`` versions of
``dr_legs_with_meshes_and_boxes.usda`` and prints a structured report of
prim, attribute, relationship, and metadata differences.

Command::

    python newton/examples/kamino/compare_dr_legs_usd_models.py
    python newton/examples/kamino/compare_dr_legs_usd_models.py --physics
    python newton/examples/kamino/compare_dr_legs_usd_models.py --old path/to/old.usda --new path/to/new.usda
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OLD = SCRIPT_DIR / "model_old" / "dr_legs_with_meshes_and_boxes.usda"
DEFAULT_NEW = SCRIPT_DIR / "model_new" / "dr_legs_with_meshes_and_boxes.usda"

SIGNIFICANT_MASS_DELTA_PCT = 1.0
SIGNIFICANT_INERTIA_DELTA_PCT = 1.0
SIGNIFICANT_JOINT_POS_M = 1.0e-3
SIGNIFICANT_JOINT_ROT_DEG = 1.0
SIGNIFICANT_BODY_POS_M = 1.0e-4

PHYSICS_BODY_ATTRS = (
    "physics:mass",
    "physics:centerOfMass",
    "physics:diagonalInertia",
    "physics:principalAxes",
    "xformOp:translate",
    "xformOp:rotateXYZ",
)

PHYSICS_JOINT_ATTRS = (
    "physics:lowerLimit",
    "physics:upperLimit",
    "physics:axis",
    "physics:localPos0",
    "physics:localPos1",
    "physics:localRot0",
    "physics:localRot1",
    "physxJoint:armature",
    "physxJoint:jointFriction",
    "drive:angular:physics:damping",
    "drive:angular:physics:stiffness",
    "drive:angular:physics:maxForce",
    "drive:angular:physics:maxVelocity",
)

PHYSICS_JOINT_RELS = ("physics:body0", "physics:body1")

KAMINO_SCENE_ATTRS = (
    "newton:kamino:constraints:alpha",
    "newton:kamino:constraints:beta",
    "newton:kamino:constraints:gamma",
    "newton:kamino:padmm:complementarityTolerance",
    "newton:kamino:padmm:dualTolerance",
    "newton:kamino:padmm:primalTolerance",
)

FLOAT_ABS_TOL = 1.0e-9
FLOAT_REL_TOL = 1.0e-6


@dataclass
class PhysicsDiffReport:
    mass_inertia: list[str] = field(default_factory=list)
    body_poses: list[str] = field(default_factory=list)
    joint_limits: list[str] = field(default_factory=list)
    joint_attachments: list[str] = field(default_factory=list)
    joint_frames: list[str] = field(default_factory=list)
    joint_global_positions: list[str] = field(default_factory=list)
    joint_dynamics: list[str] = field(default_factory=list)
    scene_settings: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)

    @property
    def difference_count(self) -> int:
        return sum(
            len(getattr(self, name))
            for name in (
                "mass_inertia",
                "body_poses",
                "joint_limits",
                "joint_attachments",
                "joint_frames",
                "joint_global_positions",
                "joint_dynamics",
                "scene_settings",
            )
        )

    def is_empty(self) -> bool:
        return self.difference_count == 0


@dataclass
class PrimSnapshot:
    path: str
    type_name: str
    api_schemas: tuple[str, ...]
    attributes: dict[str, Any]
    relationships: dict[str, tuple[str, ...]]
    metadata: dict[str, Any]


@dataclass
class DiffReport:
    stage_metadata: list[str] = field(default_factory=list)
    only_in_old: list[str] = field(default_factory=list)
    only_in_new: list[str] = field(default_factory=list)
    type_mismatches: list[str] = field(default_factory=list)
    api_schema_diffs: list[str] = field(default_factory=list)
    attribute_diffs: list[str] = field(default_factory=list)
    relationship_diffs: list[str] = field(default_factory=list)
    metadata_diffs: list[str] = field(default_factory=list)

    @property
    def difference_count(self) -> int:
        return sum(
            len(getattr(self, name))
            for name in (
                "stage_metadata",
                "only_in_old",
                "only_in_new",
                "type_mismatches",
                "api_schema_diffs",
                "attribute_diffs",
                "relationship_diffs",
                "metadata_diffs",
            )
        )

    def is_empty(self) -> bool:
        return self.difference_count == 0


def _format_value(value: Any) -> str:
    if isinstance(value, Sdf.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return repr(value)
    return repr(value)


def _values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        # USD may return Gf types; compare via string for simple scalars.
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return _floats_equal(float(left), float(right))
        return str(left) == str(right)

    if isinstance(left, float):
        return _floats_equal(left, right)

    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right, strict=True))

    return left == right


def _floats_equal(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)


def _collect_stage_metadata(stage: Usd.Stage) -> dict[str, Any]:
    pseudo_root = stage.GetRootLayer().pseudoRoot
    metadata: dict[str, Any] = {}
    for key in pseudo_root.ListInfoKeys():
        metadata[key] = pseudo_root.GetInfo(key)
    return metadata


def _collect_prim_snapshot(prim: Usd.Prim) -> PrimSnapshot:
    attributes: dict[str, Any] = {}
    for attr in prim.GetAttributes():
        if not attr.HasAuthoredValue():
            continue
        attributes[attr.GetName()] = attr.Get()

    relationships: dict[str, tuple[str, ...]] = {}
    for rel in prim.GetRelationships():
        if not rel.HasAuthoredTargets():
            continue
        relationships[rel.GetName()] = tuple(str(target) for target in rel.GetTargets())

    metadata: dict[str, Any] = {}
    for key in prim.GetAllMetadata():
        metadata[key] = prim.GetMetadata(key)

    api_schemas = tuple(sorted(prim.GetAppliedSchemas()))
    return PrimSnapshot(
        path=str(prim.GetPath()),
        type_name=prim.GetTypeName(),
        api_schemas=api_schemas,
        attributes=attributes,
        relationships=relationships,
        metadata=metadata,
    )


def _collect_prim_snapshots(stage: Usd.Stage) -> dict[str, PrimSnapshot]:
    snapshots: dict[str, PrimSnapshot] = {}
    for prim in stage.Traverse():
        snapshots[str(prim.GetPath())] = _collect_prim_snapshot(prim)
    return snapshots


def _compare_stage_metadata(old_meta: dict[str, Any], new_meta: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    keys = sorted(set(old_meta) | set(new_meta))
    for key in keys:
        old_value = old_meta.get(key, "<missing>")
        new_value = new_meta.get(key, "<missing>")
        if not _values_equal(old_value, new_value):
            diffs.append(f"  {key}: {_format_value(old_value)} -> {_format_value(new_value)}")
    return diffs


@dataclass
class ComparisonResult:
    raw: DiffReport
    normalized: DiffReport
    physics: PhysicsDiffReport


def _normalize_usd_path(path: str) -> str:
    """Map legacy hierarchy names onto the new layout for semantic comparison."""
    parts = [part for part in path.split("/") if part]
    normalized: list[str] = []
    for index, part in enumerate(parts):
        if part.lower() == "dr_legs":
            normalized.append("dr_legs")
        elif part == "RigidBodies" and index < len(parts) - 1:
            continue
        else:
            normalized.append(part)
    return "/" + "/".join(normalized) if normalized else "/"


def _normalize_snapshot(snapshot: PrimSnapshot) -> PrimSnapshot:
    normalized_attributes: dict[str, Any] = {}
    for name, value in snapshot.attributes.items():
        if isinstance(value, Sdf.Path):
            normalized_attributes[name] = _normalize_usd_path(str(value))
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], Sdf.Path):
            normalized_attributes[name] = [_normalize_usd_path(str(item)) for item in value]
        else:
            normalized_attributes[name] = value

    normalized_relationships = {
        name: tuple(_normalize_usd_path(target) for target in targets)
        for name, targets in snapshot.relationships.items()
    }

    return PrimSnapshot(
        path=_normalize_usd_path(snapshot.path),
        type_name=snapshot.type_name,
        api_schemas=snapshot.api_schemas,
        attributes=normalized_attributes,
        relationships=normalized_relationships,
        metadata=snapshot.metadata,
    )


def _rekey_snapshots(snapshots: dict[str, PrimSnapshot], *, normalize: bool) -> dict[str, PrimSnapshot]:
    rekeyed: dict[str, PrimSnapshot] = {}
    for snapshot in snapshots.values():
        key = _normalize_usd_path(snapshot.path) if normalize else snapshot.path
        if key in rekeyed:
            raise RuntimeError(
                f"Duplicate {'normalized ' if normalize else ''}prim path {key!r} while comparing models."
            )
        rekeyed[key] = _normalize_snapshot(snapshot) if normalize else snapshot
    return rekeyed


def _compare_prim_collections(old_prims: dict[str, PrimSnapshot], new_prims: dict[str, PrimSnapshot]) -> DiffReport:
    report = DiffReport()

    old_paths = set(old_prims)
    new_paths = set(new_prims)
    report.only_in_old = sorted(old_paths - new_paths)
    report.only_in_new = sorted(new_paths - old_paths)

    for path in sorted(old_paths & new_paths):
        prim_report = _compare_prim_snapshots(old_prims[path], new_prims[path])
        report.type_mismatches.extend(prim_report.type_mismatches)
        report.api_schema_diffs.extend(prim_report.api_schema_diffs)
        report.attribute_diffs.extend(prim_report.attribute_diffs)
        report.relationship_diffs.extend(prim_report.relationship_diffs)
        report.metadata_diffs.extend(prim_report.metadata_diffs)

    return report


def _compare_prim_snapshots(old: PrimSnapshot, new: PrimSnapshot) -> DiffReport:
    report = DiffReport()

    if old.type_name != new.type_name:
        report.type_mismatches.append(f"  {old.path}: {old.type_name} -> {new.type_name}")

    if old.api_schemas != new.api_schemas:
        report.api_schema_diffs.append(
            f"  {old.path}: {_format_value(old.api_schemas)} -> {_format_value(new.api_schemas)}"
        )

    attr_names = sorted(set(old.attributes) | set(new.attributes))
    for name in attr_names:
        old_value = old.attributes.get(name, "<missing>")
        new_value = new.attributes.get(name, "<missing>")
        if not _values_equal(old_value, new_value):
            report.attribute_diffs.append(
                f"  {old.path}.{name}: {_format_value(old_value)} -> {_format_value(new_value)}"
            )

    rel_names = sorted(set(old.relationships) | set(new.relationships))
    for name in rel_names:
        old_targets = old.relationships.get(name, ())
        new_targets = new.relationships.get(name, ())
        if old_targets != new_targets:
            report.relationship_diffs.append(
                f"  {old.path}.{name}: {_format_value(old_targets)} -> {_format_value(new_targets)}"
            )

    meta_keys = sorted(set(old.metadata) | set(new.metadata))
    for key in meta_keys:
        old_value = old.metadata.get(key, "<missing>")
        new_value = new.metadata.get(key, "<missing>")
        if not _values_equal(old_value, new_value):
            report.metadata_diffs.append(
                f"  {old.path} [{key}]: {_format_value(old_value)} -> {_format_value(new_value)}"
            )

    return report


def _percent_delta(old_value: float, new_value: float) -> float:
    if abs(old_value) < 1e-12:
        return float("nan")
    return 100.0 * (new_value - old_value) / old_value


def _quat_angle_deg(left: Any, right: Any) -> float:
    def unit(quat: Any) -> tuple[float, float, float, float]:
        w = quat.GetReal()
        v = quat.GetImaginary()
        norm = math.sqrt(w * w + v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        return (w / norm, v[0] / norm, v[1] / norm, v[2] / norm)

    a = unit(left)
    b = unit(right)
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * math.degrees(math.acos(min(1.0, dot)))


def _vec_length_delta(left: Any, right: Any) -> float:
    if isinstance(left, (tuple, list)):
        delta = Gf.Vec3f(left) - Gf.Vec3f(right)
    else:
        delta = Gf.Vec3d(left) - Gf.Vec3d(right)
    return delta.GetLength()


def _as_vec3d(value: Any) -> Gf.Vec3d:
    if isinstance(value, (tuple, list)):
        return Gf.Vec3d(value)
    return Gf.Vec3d(value)


def _compute_joint_global_position(
    joint: PrimSnapshot,
    bodies: dict[str, PrimSnapshot],
) -> tuple[Gf.Vec3d, str] | None:
    """Return global joint position from body0 translate + physics:localPos0."""
    body_targets = joint.relationships.get("physics:body0", ())
    if not body_targets:
        return None

    body_path = body_targets[0]
    body = bodies.get(body_path)
    if body is None:
        return None

    local_pos = joint.attributes.get("physics:localPos0")
    body_translate = body.attributes.get("xformOp:translate")
    if local_pos is None or body_translate is None:
        return None

    return _as_vec3d(body_translate) + _as_vec3d(local_pos), body_path


def _is_rigid_body(snapshot: PrimSnapshot) -> bool:
    return snapshot.type_name == "Xform" and "physics:mass" in snapshot.attributes


def _is_joint(snapshot: PrimSnapshot) -> bool:
    return snapshot.type_name.endswith("Joint")


def compare_physics_properties(
    old_prims: dict[str, PrimSnapshot],
    new_prims: dict[str, PrimSnapshot],
) -> PhysicsDiffReport:
    report = PhysicsDiffReport()
    mass_total_old = 0.0
    mass_total_new = 0.0
    significant_mass = 0
    significant_inertia = 0

    bodies = sorted(
        path
        for path in set(old_prims) & set(new_prims)
        if _is_rigid_body(old_prims[path]) and _is_rigid_body(new_prims[path])
    )
    for path in bodies:
        name = path.rsplit("/", 1)[-1]
        old = old_prims[path]
        new = new_prims[path]

        old_mass = old.attributes["physics:mass"]
        new_mass = new.attributes["physics:mass"]
        mass_total_old += old_mass
        mass_total_new += new_mass

        mass_delta = _percent_delta(old_mass, new_mass)
        if not _values_equal(old_mass, new_mass):
            marker = " ***" if abs(mass_delta) >= SIGNIFICANT_MASS_DELTA_PCT else ""
            report.mass_inertia.append(
                f"  {name}.physics:mass: {old_mass:.8f} -> {new_mass:.8f} ({mass_delta:+.3f}%){marker}"
            )
            if abs(mass_delta) >= SIGNIFICANT_MASS_DELTA_PCT:
                significant_mass += 1

        for attr in ("physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"):
            old_value = old.attributes.get(attr)
            new_value = new.attributes.get(attr)
            if old_value is None and new_value is None:
                continue
            if _values_equal(old_value, new_value):
                continue
            if attr == "physics:diagonalInertia":
                deltas = [_percent_delta(old_value[i], new_value[i]) for i in range(3)]
                marker = " ***" if any(abs(delta) >= SIGNIFICANT_INERTIA_DELTA_PCT for delta in deltas) else ""
                if marker:
                    significant_inertia += 1
                report.mass_inertia.append(
                    f"  {name}.{attr}: {_format_value(old_value)} -> {_format_value(new_value)} "
                    f"(dI=[{deltas[0]:+.2f}%, {deltas[1]:+.2f}%, {deltas[2]:+.2f}%]){marker}"
                )
            else:
                report.mass_inertia.append(f"  {name}.{attr}: {_format_value(old_value)} -> {_format_value(new_value)}")

        old_translate = old.attributes.get("xformOp:translate")
        new_translate = new.attributes.get("xformOp:translate")
        if old_translate is not None and new_translate is not None:
            pos_delta = _vec_length_delta(old_translate, new_translate)
            if not _values_equal(old_translate, new_translate):
                marker = " ***" if pos_delta >= SIGNIFICANT_BODY_POS_M else ""
                report.body_poses.append(
                    f"  {name}.xformOp:translate: {_format_value(old_translate)} -> "
                    f"{_format_value(new_translate)} (|delta|={pos_delta * 1000:.3f} mm){marker}"
                )

    joints = sorted(
        path for path in set(old_prims) & set(new_prims) if _is_joint(old_prims[path]) and _is_joint(new_prims[path])
    )
    for path in joints:
        name = path.rsplit("/", 1)[-1]
        old = old_prims[path]
        new = new_prims[path]

        old_lower = old.attributes.get("physics:lowerLimit")
        old_upper = old.attributes.get("physics:upperLimit")
        new_lower = new.attributes.get("physics:lowerLimit")
        new_upper = new.attributes.get("physics:upperLimit")
        if not _values_equal(old_lower, new_lower) or not _values_equal(old_upper, new_upper):
            report.joint_limits.append(
                f"  {name}: [{_format_value(old_lower)}, {_format_value(old_upper)}] -> "
                f"[{_format_value(new_lower)}, {_format_value(new_upper)}] ***"
            )

        old_body0 = old.relationships.get("physics:body0", ())
        old_body1 = old.relationships.get("physics:body1", ())
        new_body0 = new.relationships.get("physics:body0", ())
        new_body1 = new.relationships.get("physics:body1", ())
        if old_body0 != new_body0 or old_body1 != new_body1:
            report.joint_attachments.append(
                f"  {name}: body0 {_format_value(old_body0)} -> {_format_value(new_body0)}; "
                f"body1 {_format_value(old_body1)} -> {_format_value(new_body1)} ***"
            )

        max_pos_delta = 0.0
        max_rot_delta = 0.0
        frame_lines: list[str] = []
        for attr in ("physics:localPos0", "physics:localPos1", "physics:localRot0", "physics:localRot1"):
            old_value = old.attributes.get(attr)
            new_value = new.attributes.get(attr)
            if old_value is None and new_value is None:
                continue
            if _values_equal(old_value, new_value):
                continue
            if "localPos" in attr:
                pos_delta = _vec_length_delta(old_value, new_value)
                max_pos_delta = max(max_pos_delta, pos_delta)
                frame_lines.append(
                    f"    {attr}: {_format_value(old_value)} -> {_format_value(new_value)} "
                    f"(|delta|={pos_delta * 1000:.2f} mm)"
                )
            else:
                rot_delta = _quat_angle_deg(old_value, new_value)
                max_rot_delta = max(max_rot_delta, rot_delta)
                frame_lines.append(
                    f"    {attr}: {_format_value(old_value)} -> {_format_value(new_value)} (angle={rot_delta:.2f} deg)"
                )
        if frame_lines and (max_pos_delta >= SIGNIFICANT_JOINT_POS_M or max_rot_delta >= SIGNIFICANT_JOINT_ROT_DEG):
            marker = " ***" if max_pos_delta >= SIGNIFICANT_JOINT_POS_M else ""
            report.joint_frames.append(
                f"  {name} (max pos {max_pos_delta * 1000:.2f} mm, max rot {max_rot_delta:.2f} deg){marker}:"
            )
            report.joint_frames.extend(frame_lines)

        dynamic_lines: list[str] = []
        for attr in PHYSICS_JOINT_ATTRS:
            if attr.startswith(("physics:local", "physics:lower", "physics:upper", "physics:axis")):
                continue
            old_value = old.attributes.get(attr)
            new_value = new.attributes.get(attr)
            if old_value is None and new_value is None:
                continue
            if not _values_equal(old_value, new_value):
                dynamic_lines.append(f"    {attr}: {_format_value(old_value)} -> {_format_value(new_value)}")
        if dynamic_lines:
            report.joint_dynamics.append(f"  {name}:")
            report.joint_dynamics.extend(dynamic_lines)

        old_global = _compute_joint_global_position(old, old_prims)
        new_global = _compute_joint_global_position(new, new_prims)
        if old_global is None or new_global is None:
            continue

        old_pos, old_body_path = old_global
        new_pos, new_body_path = new_global
        if _values_equal(old_pos, new_pos):
            continue

        pos_delta = (old_pos - new_pos).GetLength()
        marker = " ***" if pos_delta >= SIGNIFICANT_JOINT_POS_M else ""
        report.joint_global_positions.append(
            f"  {name} (via {old_body_path.rsplit('/', 1)[-1]}): "
            f"{_format_value(old_pos)} -> {_format_value(new_pos)} "
            f"(|delta|={pos_delta * 1000:.3f} mm){marker}"
        )
        if old_body_path != new_body_path:
            report.joint_global_positions.append(
                f"    body0 path: {_format_value(old_body_path)} -> {_format_value(new_body_path)}"
            )

    old_scene = old_prims.get("/PhysicsScene")
    new_scene = new_prims.get("/PhysicsScene")
    if old_scene and new_scene:
        for attr in KAMINO_SCENE_ATTRS:
            old_value = old_scene.attributes.get(attr)
            new_value = new_scene.attributes.get(attr)
            if not _values_equal(old_value, new_value):
                report.scene_settings.append(f"  {attr}: {_format_value(old_value)} -> {_format_value(new_value)}")

    total_mass_delta = _percent_delta(mass_total_old, mass_total_new)
    report.summary = [
        f"  Total rigid-body mass: {mass_total_old:.6f} kg -> {mass_total_new:.6f} kg ({total_mass_delta:+.3f}%)",
        f"  Bodies with >={SIGNIFICANT_MASS_DELTA_PCT:.0f}% mass change: {significant_mass}",
        f"  Bodies with >={SIGNIFICANT_INERTIA_DELTA_PCT:.0f}% inertia change: {significant_inertia}",
        f"  Joint limit changes: {len(report.joint_limits)}",
        f"  Joint body-attachment changes: {len(report.joint_attachments)}",
        f"  Joints with large frame changes: {sum(1 for line in report.joint_frames if line.startswith('  j'))}",
        f"  Joints with global position changes: {len(report.joint_global_positions)}",
        f"  Joints with drive/friction/armature changes: {sum(1 for line in report.joint_dynamics if line.endswith(':'))}",
    ]
    return report


def compare_usd_stages(old_path: Path, new_path: Path) -> ComparisonResult:
    old_stage = Usd.Stage.Open(str(old_path), Usd.Stage.LoadNone)
    new_stage = Usd.Stage.Open(str(new_path), Usd.Stage.LoadNone)
    if old_stage is None:
        raise RuntimeError(f"Failed to open USD stage: {old_path}")
    if new_stage is None:
        raise RuntimeError(f"Failed to open USD stage: {new_path}")

    stage_metadata = _compare_stage_metadata(
        _collect_stage_metadata(old_stage),
        _collect_stage_metadata(new_stage),
    )

    old_prims = _collect_prim_snapshots(old_stage)
    new_prims = _collect_prim_snapshots(new_stage)

    raw_report = _compare_prim_collections(old_prims, new_prims)
    raw_report.stage_metadata = stage_metadata

    normalized_report = _compare_prim_collections(
        _rekey_snapshots(old_prims, normalize=True),
        _rekey_snapshots(new_prims, normalize=True),
    )
    normalized_report.stage_metadata = stage_metadata

    physics_report = compare_physics_properties(
        _rekey_snapshots(old_prims, normalize=True),
        _rekey_snapshots(new_prims, normalize=True),
    )

    return ComparisonResult(raw=raw_report, normalized=normalized_report, physics=physics_report)


def _print_section(title: str, lines: Iterable[str]) -> None:
    materialized = list(lines)
    if not materialized:
        return
    print(f"\n{title} ({len(materialized)})")
    print("-" * len(title))
    for line in materialized:
        print(line)


def _print_diff_report(title: str, report: DiffReport) -> None:
    if report.is_empty():
        print(f"\n{title}: no differences.")
        return

    print(f"\n{title} ({report.difference_count} differences)")
    print("=" * len(title))
    _print_section("Stage metadata", report.stage_metadata)
    _print_section("Prims only in old model", (f"  {path}" for path in report.only_in_old))
    _print_section("Prims only in new model", (f"  {path}" for path in report.only_in_new))
    _print_section("Prim type mismatches", report.type_mismatches)
    _print_section("Applied API schema differences", report.api_schema_diffs)
    _print_section("Authored attribute differences", report.attribute_diffs)
    _print_section("Authored relationship differences", report.relationship_diffs)
    _print_section("Prim metadata differences", report.metadata_diffs)


def _print_physics_report(report: PhysicsDiffReport) -> None:
    print("\nPhysics-relevant comparison (normalized paths)")
    print("=" * 46)
    _print_section("Summary", report.summary)
    _print_section("Mass / inertia / COM / principal axes", report.mass_inertia)
    _print_section("Rigid-body root transforms", report.body_poses)
    _print_section("Joint limits", report.joint_limits)
    _print_section("Joint body attachments", report.joint_attachments)
    _print_section("Joint frames (large deltas only)", report.joint_frames)
    _print_section("Joint global positions (body0 + localPos0)", report.joint_global_positions)
    _print_section("Joint drives / friction / armature", report.joint_dynamics)
    _print_section("Kamino scene settings", report.scene_settings)
    if report.is_empty():
        print("\nNo physics-relevant differences found.")


def print_report(result: ComparisonResult, old_path: Path, new_path: Path, *, physics_only: bool) -> None:
    print(f"Comparing USD models:\n  old: {old_path}\n  new: {new_path}")

    if physics_only:
        _print_physics_report(result.physics)
        return

    if result.raw.is_empty() and result.normalized.is_empty():
        print("\nNo differences found.")
    else:
        _print_diff_report("Exact prim-path comparison", result.raw)
        _print_diff_report(
            "Normalized prim-path comparison (DR_Legs->dr_legs, RigidBodies removed)",
            result.normalized,
        )

    _print_physics_report(result.physics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the old and new DR Legs USD model variants.",
    )
    parser.add_argument(
        "--old",
        type=Path,
        default=DEFAULT_OLD,
        help=f"Path to the old USD model (default: {DEFAULT_OLD.name})",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=DEFAULT_NEW,
        help=f"Path to the new USD model (default: {DEFAULT_NEW.name})",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Print only the physics-relevant comparison report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    old_path = args.old.resolve()
    new_path = args.new.resolve()

    if not old_path.is_file():
        print(f"Old model not found: {old_path}", file=sys.stderr)
        return 1
    if not new_path.is_file():
        print(f"New model not found: {new_path}", file=sys.stderr)
        return 1

    result = compare_usd_stages(old_path, new_path)
    print_report(result, old_path, new_path, physics_only=args.physics)
    has_differences = not result.physics.is_empty() or (
        not args.physics and (not result.raw.is_empty() or not result.normalized.is_empty())
    )
    return 0 if not has_differences else 2


if __name__ == "__main__":
    raise SystemExit(main())
