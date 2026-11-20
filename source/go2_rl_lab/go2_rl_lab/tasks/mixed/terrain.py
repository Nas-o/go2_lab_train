"""Procedural mixed-terrain configuration for the Go2 mixed task.

The sub-terrains follow the Isaac Lab height-field/mesh definitions of the source
task, re-expressed through mjlab's ``SubTerrainCfg`` API: each terrain returns
MuJoCo height-field or mesh geometries instead of USD-imported trimesh meshes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput


MESH_BASE_THICKNESS = 0.10
SPAWN_CENTER_HALF_EXTENT = 1.0
"""Half extent of the centered spawn region used to derive the terrain origin height."""


def _emit_height_field(
    spec: mujoco.MjSpec,
    cfg: SubTerrainCfg,
    noise: np.ndarray,
    vertical_scale: float,
    inverted: bool = False,
    spawn_height: float | None = None,
) -> TerrainOutput:
    """Add one height-field geom for ``noise`` and return its spawn origin."""
    body = spec.body("terrain")

    elevation_min = np.min(noise)
    elevation_max = np.max(noise)
    elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1

    max_physical_height = elevation_range * vertical_scale
    if elevation_range > 0:
        normalized_elevation = (noise - elevation_min) / elevation_range
    else:
        normalized_elevation = np.zeros_like(noise, dtype=np.float64)

    unique_id = uuid.uuid4().hex
    field = spec.add_hfield(
        name=f"hfield_{unique_id}",
        size=[cfg.size[0] / 2, cfg.size[1] / 2, max_physical_height, max_physical_height],
        nrow=noise.shape[0],
        ncol=noise.shape[1],
        userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
    )

    hfield_z_offset = -max_physical_height if inverted else 0.0
    material_name = color_by_height(spec, noise, unique_id, normalized_elevation)
    hfield_geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=field.name,
        pos=[cfg.size[0] / 2, cfg.size[1] / 2, hfield_z_offset],
        material=material_name,
    )

    if spawn_height is None:
        center = (noise.shape[0] // 2, noise.shape[1] // 2)
        spawn_height = float(noise[center]) * vertical_scale
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, spawn_height + hfield_z_offset])
    return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=hfield_geom, hfield=field)])


def _wave_noise(cfg: "WaveTerrainCfg", difficulty: float, rng: np.random.Generator) -> np.ndarray:
    """Return the wave plus random-roughness height field of the source task."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    amplitude_pixels = 0.5 * amplitude / cfg.vertical_scale

    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    x = np.arange(0, width_pixels).reshape(width_pixels, 1)
    y = np.arange(0, length_pixels).reshape(1, length_pixels)
    wave = amplitude_pixels * (np.cos(y * wave_number) + np.sin(x * wave_number))

    rough = _random_uniform_noise(cfg, rng)
    return np.rint(wave + rough).astype(np.int16)


def _random_uniform_noise(cfg: "WaveTerrainCfg", rng: np.random.Generator) -> np.ndarray:
    """Return interpolated uniform roughness, matching Isaac Lab's uniform terrain."""
    if cfg.downsampled_scale is None:
        downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )
    else:
        downsampled_scale = cfg.downsampled_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    width_downsampled = int(cfg.size[0] / downsampled_scale)
    length_downsampled = int(cfg.size[1] / downsampled_scale)
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    height_range = np.arange(height_min, height_max + height_step, height_step)
    height_field_downsampled = rng.choice(height_range, size=(width_downsampled, length_downsampled))

    x = np.linspace(0, cfg.size[0], width_downsampled)
    y = np.linspace(0, cfg.size[1], length_downsampled)
    func = _rect_bivariate_spline(x, y, height_field_downsampled)
    x_upsampled = np.linspace(0, cfg.size[0], width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1], length_pixels)
    return np.rint(func(x_upsampled, y_upsampled)).astype(np.int16)


def _rect_bivariate_spline(
    x: np.ndarray, y: np.ndarray, values: np.ndarray
):
    """Return a bilinear interpolator over the downsampled height grid."""
    import scipy.interpolate as interpolate

    return interpolate.RectBivariateSpline(x, y, values)


def _pyramid_slope_noise(cfg: "RoughSlopeTerrainCfg", difficulty: float) -> np.ndarray:
    """Return the pyramid slope height field used as the rough-slope base."""
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = slope * cfg.size[0] / 2 / cfg.vertical_scale
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width_pixels, 1)
    yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length_pixels)
    hf_raw = height_max * xx * yy

    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    return np.rint(hf_raw).astype(np.int16)


def _pyramid_stairs_noise(cfg: "PyramidStairsTerrainCfg", difficulty: float) -> np.ndarray:
    """Return Isaac Lab's pyramid stair height field, including the inverted form."""
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        current_step_height += step_height
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height
    return np.rint(hf_raw).astype(np.int16)


@dataclass(kw_only=True)
class WaveTerrainCfg(SubTerrainCfg):
    """Sinusoidal waves combined with fine random roughness."""

    amplitude_range: tuple[float, float] = (0.1, 0.28)
    num_waves: int = 5
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        noise = _wave_noise(self, difficulty, rng)
        return _emit_height_field(spec, self, noise, self.vertical_scale)


@dataclass(kw_only=True)
class RoughSlopeTerrainCfg(SubTerrainCfg):
    """Pyramid slope combined with fine random roughness."""

    slope_range: tuple[float, float] = (0.1, 0.568)
    platform_width: float = 3.0
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        noise = np.rint(_pyramid_slope_noise(self, difficulty) + _random_uniform_noise(self, rng))
        return _emit_height_field(spec, self, noise.astype(np.int16), self.vertical_scale)


@dataclass(kw_only=True)
class PyramidStairsTerrainCfg(SubTerrainCfg):
    """Pyramid stairs with an optional inverted (downward) direction."""

    step_height_range: tuple[float, float] = (0.05, 0.257)
    step_width: float = 0.31
    platform_width: float = 3.0
    inverted: bool = False
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng
        noise = _pyramid_stairs_noise(self, difficulty)
        return _emit_height_field(spec, self, noise, self.vertical_scale, inverted=self.inverted)


@dataclass(kw_only=True)
class WheelsTerrainCfg(SubTerrainCfg):
    """Field of flat tyres around a tyre-free spawn disc.

    MuJoCo only collides convex geoms, so a merged tyre mesh would be replaced
    by its convex hull and turn the whole tile into a slab. Each tyre is
    therefore built from ``ring_segments`` convex box wedges that keep the inner
    hole and never trigger the EPA polytope warning.
    """

    outer_radius: float = 0.30
    """Tyre outer radius in meters (0.6 m diameter)."""

    inner_radius_range: tuple[float, float] = (0.05, 0.25)
    """Tyre hole radius at difficulty 0 and 1; grows with the difficulty level."""

    easy_height_range: tuple[float, float] = (0.01, 0.03)
    """Per-tyre height range at difficulty 0."""

    height_range: tuple[float, float] = (0.15, 0.17)
    """Per-tyre height range at difficulty 1; interpolated from ``easy_height_range``."""

    max_gap: float = 0.0
    """Extra center-to-center gap at difficulty 1; 0 keeps neighbouring tyres in contact."""

    clear_radius: float = 1.0
    """Radius of the tyre-free disc around the terrain center."""

    ring_segments: int = 10
    """Number of convex box wedges used to approximate one tyre ring."""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        level = float(np.clip(difficulty, 0.0, 1.0))
        size_x, size_y = (float(value) for value in self.size)
        center = np.array((size_x * 0.5, size_y * 0.5))
        body = spec.body("terrain")

        ground = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size_x / 2, size_y / 2, MESH_BASE_THICKNESS / 2],
            pos=[center[0], center[1], -MESH_BASE_THICKNESS / 2],
        )
        geometries = [TerrainGeometry(geom=ground)]

        inner_radius = float(np.interp(level, (0.0, 1.0), self.inner_radius_range))
        height_range = (1.0 - level) * np.asarray(self.easy_height_range) + level * np.asarray(
            self.height_range
        )
        pitch = 2.0 * self.outer_radius + level * self.max_gap
        skip_radius = self.clear_radius + self.outer_radius
        origin_height = 0.0
        num_cells = int(np.ceil(max(size_x, size_y) / pitch)) + 1

        ring_mid_radius = 0.5 * (inner_radius + self.outer_radius)
        radial_half = 0.5 * (self.outer_radius - inner_radius)
        tangential_half = self.outer_radius * np.sin(np.pi / self.ring_segments)

        for row in range(-num_cells, num_cells + 1):
            for col in range(-num_cells, num_cells + 1):
                offset = np.array(((row + 0.5) * pitch, (col + 0.5) * pitch))
                if np.linalg.norm(offset) < skip_radius:
                    continue
                position = center + offset
                if not (
                    self.outer_radius <= position[0] <= size_x - self.outer_radius
                    and self.outer_radius <= position[1] <= size_y - self.outer_radius
                ):
                    continue
                height = float(rng.uniform(*height_range))
                for segment in range(self.ring_segments):
                    angle = 2.0 * np.pi * (segment + 0.5) / self.ring_segments
                    wedge = body.add_geom(
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[tangential_half, radial_half, height / 2],
                        pos=[
                            position[0] + ring_mid_radius * np.cos(angle),
                            position[1] + ring_mid_radius * np.sin(angle),
                            height / 2,
                        ],
                        quat=[np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
                    )
                    geometries.append(TerrainGeometry(geom=wedge))
                if (
                    abs(position[0] - center[0]) <= SPAWN_CENTER_HALF_EXTENT + self.outer_radius
                    and abs(position[1] - center[1]) <= SPAWN_CENTER_HALF_EXTENT + self.outer_radius
                ):
                    origin_height = max(origin_height, height)

        origin = np.array((center[0], center[1], origin_height))
        return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class SmallSquaresTerrainCfg(SubTerrainCfg):
    """Block terrain of randomly raised square columns."""

    square_size: float = 0.1
    """Edge length of every square column in meters."""

    height_range: tuple[float, float] = (0.01, 0.17)
    """Maximum square height at difficulty 0 and 1; heights are uniform in ``[0, x]``."""

    horizontal_scale: float = 0.05
    """Height-field sampling used to keep column tops flat."""

    vertical_scale: float = 0.005

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        level = float(np.clip(difficulty, 0.0, 1.0))
        size_x, size_y = (float(value) for value in self.size)
        max_height = float(np.interp(level, (0.0, 1.0), self.height_range))
        num_rows = int(round(size_x / self.square_size))
        num_cols = int(round(size_y / self.square_size))
        heights = rng.uniform(0.0, max_height, size=(num_rows, num_cols))

        # Each square keeps a flat top: both height-field samples inside a cell
        # carry the cell height, so only the cell seams form short ramps.
        samples_per_cell = int(round(self.square_size / self.horizontal_scale))
        noise = np.rint(
            np.repeat(np.repeat(heights, samples_per_cell, axis=0), samples_per_cell, axis=1)
            / self.vertical_scale
        ).astype(np.int16)

        half = max(1, int(round(SPAWN_CENTER_HALF_EXTENT / self.square_size)))
        center_row, center_col = num_rows // 2, num_cols // 2
        center_heights = heights[
            max(0, center_row - half) : min(num_rows, center_row + half),
            max(0, center_col - half) : min(num_cols, center_col + half),
        ]
        origin_height = float(np.max(center_heights))
        return _emit_height_field(
            spec, self, noise, self.vertical_scale, spawn_height=origin_height
        )


# Shared terrain configuration for the mixed-terrain task.
MIXED_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(9.0, 9.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
    sub_terrains={
        "wave": WaveTerrainCfg(proportion=0.05),
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.1, 0.568),
            platform_width=3.0,
            inverted=True,
        ),
        "slope_down": HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.1, 0.568),
            platform_width=3.0,
        ),
        "rough_slope": RoughSlopeTerrainCfg(proportion=0.05),
        "stairs_up": PyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.257),
            step_width=0.31,
            platform_width=3.0,
            inverted=True,
        ),
        "stairs_down": PyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.257),
            step_width=0.31,
            platform_width=3.0,
        ),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.275),
            num_obstacles=20,
            platform_width=3.0,
        ),
        "flat": BoxFlatTerrainCfg(proportion=0.15),
        "wheels": WheelsTerrainCfg(proportion=0.05),
        "small_squares": SmallSquaresTerrainCfg(proportion=0.05),
    },
)


__all__ = [
    "MIXED_TERRAIN_CFG",
    "PyramidStairsTerrainCfg",
    "RoughSlopeTerrainCfg",
    "SmallSquaresTerrainCfg",
    "WaveTerrainCfg",
    "WheelsTerrainCfg",
]
