from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from .pdb import Atom, coords_array
from .surface import SurfaceConfig, compute_density


DEFAULT_CHIMERAX_BIN = "/Applications/ChimeraX-1.11.app/Contents/MacOS/ChimeraX"


def find_chimerax(default: str | None = None) -> str:
    """Return a ChimeraX executable path."""

    candidates = [
        default,
        shutil.which("chimerax"),
        shutil.which("ChimeraX"),
        DEFAULT_CHIMERAX_BIN,
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("ChimeraX executable not found. Pass --chimerax-bin explicitly.")


def _quote_chimerax_path(path: Path) -> str:
    text = str(path.resolve())
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def run_chimerax_surface(
    pdb_path: str | Path,
    obj_path: str | Path,
    *,
    chimerax_bin: str | None = None,
) -> Path:
    """Generate a solvent-excluded molecular surface OBJ with ChimeraX."""

    executable = find_chimerax(chimerax_bin)
    pdb_path = Path(pdb_path)
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "--nogui",
        "--silent",
        "--cmd",
        f"open {_quote_chimerax_path(pdb_path)}",
        "--cmd",
        "surface",
        "--cmd",
        f"save {_quote_chimerax_path(obj_path)} format obj",
        "--cmd",
        "exit",
    ]
    subprocess.run(command, check=True)
    return obj_path


def read_obj_vertices_normals(obj_path: str | Path, *, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Read OBJ vertices and normals from ChimeraX output."""

    if stride < 1:
        raise ValueError("stride must be >= 1")

    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []

    with Path(obj_path).open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("vn "):
                _, nx, ny, nz, *_ = line.split()
                normals.append((float(nx), float(ny), float(nz)))

    vertex_array = np.array(vertices, dtype=float)
    normal_array = np.array(normals, dtype=float)
    if len(normal_array) != len(vertex_array):
        normal_array = np.full_like(vertex_array, np.nan)

    if stride > 1:
        vertex_array = vertex_array[::stride]
        normal_array = normal_array[::stride]

    return vertex_array, normal_array


def _residue_label(atom: Atom | None) -> str:
    if atom is None:
        return ""
    residue_number = str(atom.residue_number)
    if atom.insertion_code:
        residue_number = f"{residue_number}{atom.insertion_code}"
    return f"{atom.chain_id}_{residue_number}_{atom.residue_name}"


def _nearest_atom_indices(points: np.ndarray, atoms: list[Atom], *, chunk_size: int = 2048) -> np.ndarray:
    if len(points) == 0 or not atoms:
        return np.full(len(points), -1, dtype=int)

    atom_coords = coords_array(atoms)
    nearest = np.empty(len(points), dtype=int)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        delta = points[start:stop, None, :] - atom_coords[None, :, :]
        distances_sq = np.einsum("ijk,ijk->ij", delta, delta)
        nearest[start:stop] = np.argmin(distances_sq, axis=1)
    return nearest


def zernike_surface_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the minimal surface columns used by Zernike2D tutorials."""

    return df.loc[:, ["res", "x", "y", "z", "nx", "ny", "nz"]]


def surface_row_residue_key(row: pd.Series) -> tuple[str, int, str, str] | None:
    """Extract a Task 1 residue key from one ChimeraX surface CSV row."""

    try:
        chain_id = row.get("chain_id")
        residue_number = row.get("residue_number")
        insertion_code = row.get("insertion_code")
        residue_name = row.get("residue_name")

        if pd.isna(chain_id) or pd.isna(residue_number) or pd.isna(residue_name):
            return None

        chain = str(chain_id).strip() or "_"
        number = int(float(residue_number))
        insertion = "" if pd.isna(insertion_code) else str(insertion_code).strip()
        residue = str(residue_name).strip()
        if not residue:
            return None
        return (chain, number, insertion, residue)
    except (TypeError, ValueError):
        return None


def filter_surface_by_residue_keys(
    df: pd.DataFrame,
    residues: set[tuple[str, int, str, str]] | frozenset[tuple[str, int, str, str]],
) -> pd.DataFrame:
    """Filter a ChimeraX full-chain surface to rows assigned to interface residues."""

    required = {"chain_id", "residue_number", "insertion_code", "residue_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Cannot filter surface by residues; missing columns: {sorted(missing)}")

    residue_set = set(residues)
    mask = [surface_row_residue_key(row) in residue_set for _, row in df.iterrows()]
    return df.loc[mask].copy().reset_index(drop=True)


def obj_surface_to_csv(
    obj_path: str | Path,
    csv_path: str | Path,
    *,
    density_atoms: list[Atom],
    config: SurfaceConfig | None = None,
    stride: int = 1,
    zernike_compatible: bool = False,
) -> pd.DataFrame:
    """Convert a ChimeraX OBJ surface to the same CSV schema used by Task 1."""

    config = config or SurfaceConfig()
    vertices, normals = read_obj_vertices_normals(obj_path, stride=stride)
    nearest_indices = _nearest_atom_indices(vertices, density_atoms)
    nearest_atoms = [
        density_atoms[int(index)] if index >= 0 else None
        for index in nearest_indices
    ]
    densities = compute_density(
        vertices,
        density_atoms,
        coords_array(density_atoms),
        config.density_alpha,
        config.density_cutoff_angstrom,
    )

    df = pd.DataFrame(
        {
            "res": [_residue_label(atom) for atom in nearest_atoms],
            "x": vertices[:, 0] if len(vertices) else [],
            "y": vertices[:, 1] if len(vertices) else [],
            "z": vertices[:, 2] if len(vertices) else [],
            "nx": normals[:, 0] if len(normals) else [],
            "ny": normals[:, 1] if len(normals) else [],
            "nz": normals[:, 2] if len(normals) else [],
            "density": densities,
            "source_atom_serial": [atom.serial if atom else pd.NA for atom in nearest_atoms],
            "chain_id": [atom.chain_id if atom else pd.NA for atom in nearest_atoms],
            "residue_number": [atom.residue_number if atom else pd.NA for atom in nearest_atoms],
            "insertion_code": [atom.insertion_code if atom else pd.NA for atom in nearest_atoms],
            "residue_name": [atom.residue_name if atom else pd.NA for atom in nearest_atoms],
            "atom_name": [atom.name if atom else pd.NA for atom in nearest_atoms],
            "element": [atom.element if atom else pd.NA for atom in nearest_atoms],
        }
    )
    if zernike_compatible:
        df = zernike_surface_columns(df)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
