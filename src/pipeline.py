from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from .chimerax import filter_surface_by_residue_keys, obj_surface_to_csv, run_chimerax_surface
from .dataset import ComplexRecord, read_complex_records
from .interface import atoms_in_residues, find_interface
from .pdb import download_pdb, read_pdb, select_chains, write_pdb
from .surface import SurfaceConfig, generate_surface_patch


@dataclass(frozen=True)
class PipelineConfig:
    dataset_path: Path = Path("data/Affinity Benchmark v5.5.xlsx")
    pdb_dir: Path = Path("data/pdb")
    output_dir: Path = Path("outputs/task1")
    download_missing: bool = False
    overwrite_downloads: bool = False
    download_verify_ssl: bool = True
    contact_cutoff_angstrom: float = 5.0
    surface: SurfaceConfig = SurfaceConfig()
    surface_backend: Literal["sampler", "chimerax", "both"] = "chimerax"
    chimerax_bin: str | None = None
    chimerax_csv_stride: int = 1
    chimerax_full_surface_filter: bool = True
    dimers_only: bool = False


def _residue_rows(residues: Iterable[tuple[str, int, str, str]]) -> list[dict[str, object]]:
    return [
        {
            "chain_id": chain,
            "residue_number": residue_number,
            "insertion_code": insertion_code,
            "residue_name": residue_name,
        }
        for chain, residue_number, insertion_code, residue_name in sorted(
            residues, key=lambda r: (r[0], r[1], r[2], r[3])
        )
    ]


def _contact_rows(contacts: Iterable[object]) -> list[dict[str, object]]:
    return [asdict(contact) for contact in contacts]


def _record_output_dir(base: Path, record: ComplexRecord) -> Path:
    return base / f"{record.row_index:05d}_{record.pdb_id}"


def _structure_path(record: ComplexRecord, config: PipelineConfig) -> Path:
    path = config.pdb_dir / f"{record.pdb_id}.pdb"
    if path.exists():
        return path
    if not config.download_missing:
        raise FileNotFoundError(
            f"Missing {path}. Re-run with --download or place {record.pdb_id}.pdb in {config.pdb_dir}."
        )
    return download_pdb(
        record.pdb_id,
        config.pdb_dir,
        overwrite=config.overwrite_downloads,
        verify_ssl=config.download_verify_ssl,
    )


def _write_chimerax_patch_surfaces_legacy(
    record_dir: Path,
    ligand_patch_atoms: list,
    receptor_patch_atoms: list,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Legacy ChimeraX mode: surface the already-cut interface patch."""

    ligand_obj = record_dir / "ligand_surface_patch_chimerax.obj"
    receptor_obj = record_dir / "receptor_surface_patch_chimerax.obj"
    run_chimerax_surface(
        record_dir / "ligand_interface_patch.pdb",
        ligand_obj,
        chimerax_bin=config.chimerax_bin,
    )
    run_chimerax_surface(
        record_dir / "receptor_interface_patch.pdb",
        receptor_obj,
        chimerax_bin=config.chimerax_bin,
    )
    ligand_surface = obj_surface_to_csv(
        ligand_obj,
        record_dir / "ligand_surface_patch_chimerax.csv",
        density_atoms=ligand_patch_atoms,
        config=config.surface,
        stride=config.chimerax_csv_stride,
    )
    receptor_surface = obj_surface_to_csv(
        receptor_obj,
        record_dir / "receptor_surface_patch_chimerax.csv",
        density_atoms=receptor_patch_atoms,
        config=config.surface,
        stride=config.chimerax_csv_stride,
    )
    ligand_surface.to_csv(record_dir / "ligand_surface_patch.csv", index=False)
    receptor_surface.to_csv(record_dir / "receptor_surface_patch.csv", index=False)
    return ligand_surface, receptor_surface, {
        "ligand_full_surface_points": len(ligand_surface),
        "receptor_full_surface_points": len(receptor_surface),
        "ligand_surface_points": len(ligand_surface),
        "receptor_surface_points": len(receptor_surface),
    }


def _write_chimerax_full_and_filtered_surfaces(
    record_dir: Path,
    ligand_atoms: list,
    receptor_atoms: list,
    interface,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Generate full-chain ChimeraX surfaces, then filter to interface residues."""

    ligand_full_obj = record_dir / "ligand_surface_full_chimerax.obj"
    receptor_full_obj = record_dir / "receptor_surface_full_chimerax.obj"
    ligand_full_csv = record_dir / "ligand_surface_full_chimerax.csv"
    receptor_full_csv = record_dir / "receptor_surface_full_chimerax.csv"
    ligand_patch_csv = record_dir / "ligand_surface_patch_chimerax.csv"
    receptor_patch_csv = record_dir / "receptor_surface_patch_chimerax.csv"

    run_chimerax_surface(
        record_dir / "ligand_chains.pdb",
        ligand_full_obj,
        chimerax_bin=config.chimerax_bin,
    )
    run_chimerax_surface(
        record_dir / "receptor_chains.pdb",
        receptor_full_obj,
        chimerax_bin=config.chimerax_bin,
    )

    ligand_full_surface = obj_surface_to_csv(
        ligand_full_obj,
        ligand_full_csv,
        density_atoms=ligand_atoms,
        config=config.surface,
        stride=config.chimerax_csv_stride,
    )
    receptor_full_surface = obj_surface_to_csv(
        receptor_full_obj,
        receptor_full_csv,
        density_atoms=receptor_atoms,
        config=config.surface,
        stride=config.chimerax_csv_stride,
    )

    ligand_filtered = filter_surface_by_residue_keys(ligand_full_surface, interface.ligand_residues)
    receptor_filtered = filter_surface_by_residue_keys(receptor_full_surface, interface.receptor_residues)
    ligand_filtered.to_csv(ligand_patch_csv, index=False)
    receptor_filtered.to_csv(receptor_patch_csv, index=False)

    # Canonical Task 1 surface inputs for Task 2 are ChimeraX filtered surfaces.
    ligand_filtered.to_csv(record_dir / "ligand_surface_patch.csv", index=False)
    receptor_filtered.to_csv(record_dir / "receptor_surface_patch.csv", index=False)

    return ligand_filtered, receptor_filtered, {
        "ligand_full_surface_points": len(ligand_full_surface),
        "receptor_full_surface_points": len(receptor_full_surface),
        "ligand_surface_points": len(ligand_filtered),
        "receptor_surface_points": len(receptor_filtered),
    }


def process_record(record: ComplexRecord, config: PipelineConfig) -> dict[str, object]:
    record_dir = _record_output_dir(config.output_dir, record)
    record_dir.mkdir(parents=True, exist_ok=True)

    try:
        pdb_path = _structure_path(record, config)
        atoms = read_pdb(pdb_path)
        ligand_atoms = select_chains(atoms, record.ligand_chains)
        receptor_atoms = select_chains(atoms, record.receptor_chains)
        if not ligand_atoms:
            raise ValueError(f"No ligand atoms found for chains {record.ligand_chains}")
        if not receptor_atoms:
            raise ValueError(f"No receptor atoms found for chains {record.receptor_chains}")

        interface = find_interface(
            ligand_atoms,
            receptor_atoms,
            cutoff_angstrom=config.contact_cutoff_angstrom,
        )
        ligand_patch_atoms = atoms_in_residues(ligand_atoms, interface.ligand_residues)
        receptor_patch_atoms = atoms_in_residues(receptor_atoms, interface.receptor_residues)

        write_pdb(record_dir / "ligand_chains.pdb", ligand_atoms)
        write_pdb(record_dir / "receptor_chains.pdb", receptor_atoms)
        write_pdb(record_dir / "ligand_interface_patch.pdb", ligand_patch_atoms)
        write_pdb(record_dir / "receptor_interface_patch.pdb", receptor_patch_atoms)

        ligand_surface = pd.DataFrame()
        receptor_surface = pd.DataFrame()
        surface_counts_extra: dict[str, int] = {}

        if config.surface_backend in {"sampler", "both"}:
            ligand_sampler_surface = generate_surface_patch(
                ligand_patch_atoms,
                ligand_atoms,
                config=config.surface,
            )
            receptor_sampler_surface = generate_surface_patch(
                receptor_patch_atoms,
                receptor_atoms,
                config=config.surface,
            )
            if config.surface_backend == "sampler":
                ligand_surface = ligand_sampler_surface
                receptor_surface = receptor_sampler_surface
                ligand_surface.to_csv(record_dir / "ligand_surface_patch.csv", index=False)
                receptor_surface.to_csv(record_dir / "receptor_surface_patch.csv", index=False)
            else:
                ligand_sampler_surface.to_csv(record_dir / "ligand_surface_patch_sampler.csv", index=False)
                receptor_sampler_surface.to_csv(record_dir / "receptor_surface_patch_sampler.csv", index=False)

        if config.surface_backend in {"chimerax", "both"}:
            if config.chimerax_full_surface_filter:
                ligand_surface, receptor_surface, surface_counts_extra = _write_chimerax_full_and_filtered_surfaces(
                    record_dir,
                    ligand_atoms,
                    receptor_atoms,
                    interface,
                    config,
                )
            else:
                ligand_surface, receptor_surface, surface_counts_extra = _write_chimerax_patch_surfaces_legacy(
                    record_dir,
                    ligand_patch_atoms,
                    receptor_patch_atoms,
                    config,
                )

        pd.DataFrame(_residue_rows(interface.ligand_residues)).to_csv(
            record_dir / "ligand_interface_residues.csv", index=False
        )
        pd.DataFrame(_residue_rows(interface.receptor_residues)).to_csv(
            record_dir / "receptor_interface_residues.csv", index=False
        )
        pd.DataFrame(_contact_rows(interface.contacts)).to_csv(record_dir / "interface_contacts.csv", index=False)

        metadata = {
            "status": "ok",
            "record": asdict(record),
            "pdb_path": str(pdb_path),
            "surface_backend": config.surface_backend,
            "chimerax_surface_mode": (
                "full_chain_then_filter"
                if config.surface_backend in {"chimerax", "both"} and config.chimerax_full_surface_filter
                else "interface_patch_legacy"
                if config.surface_backend in {"chimerax", "both"}
                else None
            ),
            "ligand_surface_filter": (
                "nearest_residue_in_interface"
                if config.surface_backend in {"chimerax", "both"} and config.chimerax_full_surface_filter
                else None
            ),
            "receptor_surface_filter": (
                "nearest_residue_in_interface"
                if config.surface_backend in {"chimerax", "both"} and config.chimerax_full_surface_filter
                else None
            ),
            "counts": {
                "all_atoms": len(atoms),
                "ligand_atoms": len(ligand_atoms),
                "receptor_atoms": len(receptor_atoms),
                "ligand_interface_residues": len(interface.ligand_residues),
                "receptor_interface_residues": len(interface.receptor_residues),
                "residue_contacts": len(interface.contacts),
                "ligand_patch_atoms": len(ligand_patch_atoms),
                "receptor_patch_atoms": len(receptor_patch_atoms),
                "ligand_surface_points": int(len(ligand_surface)),
                "receptor_surface_points": int(len(receptor_surface)),
                **surface_counts_extra,
            },
            "parameters": {
                "contact_cutoff_angstrom": config.contact_cutoff_angstrom,
                "dimers_only": config.dimers_only,
                "download_verify_ssl": config.download_verify_ssl,
                "surface": asdict(config.surface),
                "surface_backend": config.surface_backend,
                "chimerax_full_surface_filter": config.chimerax_full_surface_filter,
                "chimerax_bin": config.chimerax_bin,
                "chimerax_csv_stride": config.chimerax_csv_stride,
            },
        }
    except Exception as exc:  # Keep batch runs going and make failures auditable.
        metadata = {
            "status": "error",
            "record": asdict(record),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    (record_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "row_index": record.row_index,
        "pdb_id": record.pdb_id,
        "status": metadata["status"],
        "output_dir": str(record_dir),
        **metadata.get("counts", {}),
        "error": metadata.get("error"),
    }


def run_pipeline(
    config: PipelineConfig,
    *,
    pdb_ids: Iterable[str] | None = None,
    start: int = 0,
    limit: int | None = None,
) -> pd.DataFrame:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_complex_records(
        config.dataset_path,
        pdb_ids=pdb_ids,
        start=start,
        limit=limit,
        dimers_only=config.dimers_only,
    )
    summaries = [process_record(record, config) for record in records]
    manifest = pd.DataFrame(summaries)
    manifest.to_csv(config.output_dir / "manifest.csv", index=False)
    return manifest
