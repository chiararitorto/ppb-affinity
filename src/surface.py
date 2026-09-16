from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import pandas as pd

from .pdb import Atom, coords_array


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 1.80,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 2.31,
}

ELECTRON_COUNTS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "CL": 17,
    "CA": 20,
    "FE": 26,
    "ZN": 30,
    "BR": 35,
    "I": 53,
}


@dataclass(frozen=True)
class SurfaceConfig:
    samples_per_atom: int = 64
    probe_radius: float = 1.4
    occlusion_tolerance: float = 0.05
    density_alpha: float = 1.8
    density_cutoff_angstrom: float = 6.0


def element_radius(element: str) -> float:
    return VDW_RADII.get(element.upper(), 1.70)


def electron_count(element: str) -> int:
    return ELECTRON_COUNTS.get(element.upper(), 6)


@lru_cache(maxsize=32)
def fibonacci_sphere(samples: int) -> np.ndarray:
    """Deterministic unit vectors distributed on a sphere."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(max(samples - 1, 1))) * 2
        radius = math.sqrt(max(0.0, 1 - y * y))
        theta = golden_angle * i
        points.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
    return np.array(points, dtype=float)


def _atom_radii(atoms: list[Atom], probe_radius: float) -> np.ndarray:
    return np.array([element_radius(atom.element) + probe_radius for atom in atoms], dtype=float)


def compute_density(points: np.ndarray, occluder_atoms: list[Atom], occluder_coords: np.ndarray, density_alpha: float, cutoff: float) -> np.ndarray:
    if len(points) == 0 or len(occluder_atoms) == 0:
        return np.zeros(len(points), dtype=float)

    electrons = np.array([electron_count(atom.element) for atom in occluder_atoms], dtype=float)
    radii = np.array([element_radius(atom.element) for atom in occluder_atoms], dtype=float)
    densities = np.zeros(len(points), dtype=float)
    cutoff_sq = cutoff * cutoff

    for start in range(0, len(points), 256):
        stop = min(start + 256, len(points))
        delta = points[start:stop, None, :] - occluder_coords[None, :, :]
        distances_sq = np.einsum("ijk,ijk->ij", delta, delta)
        mask = distances_sq <= cutoff_sq
        scaled = distances_sq / np.maximum(radii[None, :] ** 2, 1e-6)
        contribution = electrons[None, :] * np.exp(-density_alpha * scaled)
        densities[start:stop] = np.where(mask, contribution, 0.0).sum(axis=1)

    return densities


def generate_surface_patch(
    patch_atoms: list[Atom],
    occluder_atoms: list[Atom],
    *,
    config: SurfaceConfig | None = None,
) -> pd.DataFrame:
    """Generate a density-annotated molecular surface point cloud for interface atoms.

    The implementation is intentionally dependency-light. It samples a solvent-accessible
    van der Waals/probe sphere around each interface atom, removes points buried by the
    same partner, and annotates the retained points with an approximate Gaussian electron
    density. The CSV columns match the xyz/normal style expected by downstream Zernike
    workflows.
    """

    config = config or SurfaceConfig()
    columns = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "density",
        "source_atom_serial",
        "chain_id",
        "residue_number",
        "insertion_code",
        "residue_name",
        "atom_name",
        "element",
    ]
    if not patch_atoms:
        return pd.DataFrame(columns=columns)

    occluder_coords = coords_array(occluder_atoms)
    occluder_radii = _atom_radii(occluder_atoms, config.probe_radius)
    unit_vectors = fibonacci_sphere(config.samples_per_atom)

    rows: list[dict[str, object]] = []
    for atom in patch_atoms:
        center = atom.coord
        surface_radius = element_radius(atom.element) + config.probe_radius
        candidate_points = center[None, :] + unit_vectors * surface_radius

        visible = np.ones(len(candidate_points), dtype=bool)
        if len(occluder_atoms) > 1:
            delta = candidate_points[:, None, :] - occluder_coords[None, :, :]
            distances_sq = np.einsum("ijk,ijk->ij", delta, delta)
            buried_thresholds = np.maximum(occluder_radii - config.occlusion_tolerance, 0.0) ** 2
            self_mask = np.array([other.serial == atom.serial for other in occluder_atoms], dtype=bool)
            buried = distances_sq < buried_thresholds[None, :]
            buried[:, self_mask] = False
            visible = ~buried.any(axis=1)

        if not visible.any():
            continue

        points = candidate_points[visible]
        normals = unit_vectors[visible]
        densities = compute_density(
            points,
            occluder_atoms,
            occluder_coords,
            config.density_alpha,
            config.density_cutoff_angstrom,
        )

        for point, normal, density_value in zip(points, normals, densities):
            rows.append(
                {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "nx": float(normal[0]),
                    "ny": float(normal[1]),
                    "nz": float(normal[2]),
                    "density": float(density_value),
                    "source_atom_serial": atom.serial,
                    "chain_id": atom.chain_id,
                    "residue_number": atom.residue_number,
                    "insertion_code": atom.insertion_code,
                    "residue_name": atom.residue_name,
                    "atom_name": atom.name,
                    "element": atom.element,
                }
            )

    return pd.DataFrame(rows, columns=columns)
