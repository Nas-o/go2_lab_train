"""Terrain-related helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.terrains import TerrainGeneratorCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.terrains import TerrainEntity


class TerrainPositionLookup:
    """Query current terrain categories on a centered curriculum-generated grid."""

    def __init__(self, cfg: TerrainGeneratorCfg, device: str) -> None:
        """Cache grid dimensions and the generator's exact column-to-category assignment."""
        if not cfg.curriculum:
            raise ValueError("Position-based terrain categories require curriculum-generated columns.")
        self.names = tuple(cfg.sub_terrains)
        proportions = np.array(
            [terrain.proportion for terrain in cfg.sub_terrains.values()], dtype=np.float64
        )
        if not np.isfinite(proportions).all() or (proportions < 0).any() or proportions.sum() <= 0:
            raise ValueError("Terrain proportions must be finite, nonnegative, and have a positive sum.")
        cumulative = np.cumsum(proportions / proportions.sum())
        columns = np.arange(cfg.num_cols) / cfg.num_cols + 0.001
        column_types = np.searchsorted(cumulative, columns, side="right")
        if (column_types >= len(self.names)).any():
            raise ValueError("Terrain column allocation exceeds the configured category count.")
        self.column_types = torch.as_tensor(column_types, dtype=torch.long, device=device)
        self.size = torch.tensor(cfg.size, dtype=torch.float32, device=device)
        self.shape = torch.tensor([cfg.num_rows, cfg.num_cols], device=device)
        self.minimum_xy = -0.5 * self.size * self.shape

    def query(self, positions_w: torch.Tensor) -> torch.Tensor:
        """Return one category ID per world position, with -1 outside the generated grid.

        Half-open cells assign exact seams to the next row or column. Border
        ground and nonfinite positions return -1 instead of an edge category.
        """
        cells = torch.floor((positions_w[:, :2] - self.minimum_xy) / self.size).long()
        inside = ((cells >= 0) & (cells < self.shape)).all(dim=-1)
        inside &= torch.isfinite(positions_w[:, :2]).all(dim=-1)
        columns = cells[:, 1].clamp(0, len(self.column_types) - 1)
        return torch.where(inside, self.column_types[columns], -1)


def get_terrain_position_lookup(env: ManagerBasedRlEnv) -> TerrainPositionLookup | None:
    """Cache geometric lookup metadata for generated terrain, leaving plane tasks unrestricted."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.cfg.terrain_type != "generator":
        return None
    generator = terrain.cfg.terrain_generator
    if generator is None or not generator.sub_terrains:
        return None
    lookup = getattr(terrain, "_position_lookup", None)
    if lookup is None:
        lookup = TerrainPositionLookup(generator, env.device)
        terrain._position_lookup = lookup
    return lookup


def get_terrain_masks(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Return terrain masks from each robot's current base world position."""
    lookup = get_terrain_position_lookup(env)
    if lookup is None:
        return {}
    terrain_ids = lookup.query(env.scene["robot"].data.root_link_pos_w)
    return {name: terrain_ids == index for index, name in enumerate(lookup.names)}


def get_assigned_terrain_masks(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Return cached terrain assignments for reset placement and curriculum statistics."""
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None:
        return {}

    cached_masks = getattr(terrain, "_assigned_terrain_masks", None)
    if cached_masks is not None:
        return cached_masks

    terrain_masks: dict[str, torch.Tensor] = {}
    terrain_generator = terrain.cfg.terrain_generator
    if (
        terrain.cfg.terrain_type == "generator"
        and terrain_generator is not None
        and terrain_generator.sub_terrains is not None
        and hasattr(terrain, "terrain_types")
    ):
        lookup = get_terrain_position_lookup(env)
        terrain_ids = lookup.column_types[terrain.terrain_types]
        terrain_masks = {name: terrain_ids == index for index, name in enumerate(lookup.names)}

    terrain._assigned_terrain_masks = terrain_masks
    return terrain_masks


def column_replicas(generator_cfg: TerrainGeneratorCfg) -> list[int]:
    """Number of environments assigned to each generator column in playback.

    The source importer assigns a single environment per non-flat terrain column
    and a single environment per flat column, so every authored column appears
    once before cells repeat.
    """
    return [1] * generator_cfg.num_cols


def configure_high_to_low_play_origins(env: ManagerBasedRlEnv) -> None:
    """Assign playback environments deterministically from hardest to easiest terrain.

    Mirrors the source ``HighToLowTerrainImporter`` allocation on top of the
    native mjlab terrain grid: rows descend by difficulty and columns ascend by
    terrain type, repeating the first cell when there are more environments than
    authored cells.
    """
    terrain: TerrainEntity = env.scene.terrain
    generator_cfg = terrain.cfg.terrain_generator
    if terrain.cfg.terrain_type != "generator" or generator_cfg is None:
        return

    num_rows = generator_cfg.num_rows
    if terrain.cfg.max_init_terrain_level is None:
        max_init_level = num_rows - 1
    else:
        max_init_level = min(terrain.cfg.max_init_terrain_level, num_rows - 1)

    replicas = column_replicas(generator_cfg)
    cells = torch.tensor(
        [
            (level, col)
            for level in range(max_init_level, -1, -1)
            for col, repeats in enumerate(replicas)
            for _ in range(repeats)
        ],
        device=env.device,
    )
    env_cells = cells[torch.arange(env.num_envs, device=env.device) % cells.shape[0]]
    terrain.terrain_levels = env_cells[:, 0]
    terrain.terrain_types = env_cells[:, 1]
    # ``Scene.env_origins`` forwards to the terrain entity, so assigning here is
    # enough for reset events and the command term.
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def configure_play_sub_terrains(
    terrain_generator: TerrainGeneratorCfg,
    proportions: Mapping[str, float] | None = None,
) -> None:
    """Restrict playback terrain generation to the requested sub-terrain proportions.

    Curriculum generation assigns sub-terrains to grid columns through normalized
    cumulative proportions over every entry of ``terrain_generator.sub_terrains``,
    so unselected terrains are zeroed out instead of removed and columns keep the
    sub-terrain definition order. Passing ``None`` keeps every sub-terrain with one
    column each, while a mapping keeps only its entries and quantizes the relative
    weights into whole columns, for example ``{"flat": 0.75, "wheels": 0.25}``
    yields three flat columns followed by one wheels column.

    Args:
        terrain_generator: Terrain generator configuration modified in place.
        proportions: Optional sub-terrain name to relative weight mapping.

    Raises:
        ValueError: If a name is unknown or no weight is positive.
    """
    sub_terrains = terrain_generator.sub_terrains
    if sub_terrains is None:
        return
    if proportions is None:
        proportions = {name: 1.0 for name in sub_terrains}

    unknown_names = set(proportions).difference(sub_terrains)
    if unknown_names:
        raise ValueError(
            f"Unknown sub-terrain names {sorted(unknown_names)}; "
            f"available names are {sorted(sub_terrains)}."
        )
    weights: dict[str, float] = {}
    for name, weight in proportions.items():
        value = float(weight)
        if value > 0.0:
            weights[name] = value
    if not weights:
        raise ValueError("At least one sub-terrain proportion must be positive.")

    smallest_weight = min(weights.values())
    column_counts = {
        name: max(1, round(weight / smallest_weight)) for name, weight in weights.items()
    }
    total_columns = sum(column_counts.values())
    terrain_generator.num_cols = total_columns
    for name, sub_terrain_cfg in sub_terrains.items():
        column_count = column_counts.get(name, 0)
        sub_terrain_cfg.proportion = column_count / total_columns if column_count else 0.0


__all__ = [
    "TerrainPositionLookup",
    "column_replicas",
    "configure_high_to_low_play_origins",
    "configure_play_sub_terrains",
    "get_assigned_terrain_masks",
    "get_terrain_masks",
    "get_terrain_position_lookup",
]
