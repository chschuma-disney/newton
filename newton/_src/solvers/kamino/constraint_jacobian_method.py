# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constraint Jacobian assembly method selection."""

from __future__ import annotations

from enum import IntEnum


class ConstraintJacobianMethod(IntEnum):
    """Method used to assemble joint constraint and actuation Jacobians."""

    GEOMETRIC = 0
    """Screw-based assembly via ``W @ R_X_bar`` wrench transforms."""

    ANALYTIC = 1
    """Closed-form per-axis screw formulas (alternative to geometric assembly)."""

    @classmethod
    def from_string(cls, s: str) -> ConstraintJacobianMethod:
        """Converts a string to a :class:`ConstraintJacobianMethod` enum value."""
        try:
            return cls[s.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid ConstraintJacobianMethod: {s}. Valid options are: {[m.name.lower() for m in cls]}"
            ) from e
