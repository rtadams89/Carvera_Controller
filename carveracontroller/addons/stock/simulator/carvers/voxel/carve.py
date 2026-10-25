"""Voxel occupancy carving from toolpath segments."""

from __future__ import annotations

from carveracontroller.addons.stock.simulator.carver_select import resolve_cutting_profile
from carveracontroller.addons.tool_visualization.tool_definition import ToolDefinition

from .grid import ChunkedVoxelGrid


def carve_segment_into_grid(
    grid: ChunkedVoxelGrid,
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    tool_def: ToolDefinition | None,
    tool_unit_scale: float = 1.0,
    *,
    a0: float = 0.0,
    a1: float = 0.0,
    profile: list[tuple[float, float]] | None = None,
) -> set[tuple[int, int, int]]:
    """Carve one toolpath segment into ``grid``.

    Returns chunks whose occupancy changed. The cutting body is the upright
    surface of revolution from the tool profile. ``a0`` / ``a1`` are A-axis
    angles in degrees; subdivision of large ``|ΔA|`` happens inside the native
    kernel.
    """
    if profile is None:
        profile = resolve_cutting_profile(tool_def, tool_unit_scale=tool_unit_scale)
    if not profile:
        return set()
    from carveracontroller.addons.stock.simulator.native import carve_voxels

    return carve_voxels(grid, [(p0, p1, float(a0), float(a1))], profile)
