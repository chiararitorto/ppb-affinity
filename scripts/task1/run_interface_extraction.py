from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Rende importabile il package src/ dalla root del repository
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline import PipelineConfig, run_pipeline
from src.surface import SurfaceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 1: read PDB IDs/chains from the PPB-Affinity Excel file, then extract "
            "binding-interface residues and molecular surface patches."
        )
    )
    parser.add_argument("--dataset", default="data/Affinity Benchmark v5.5.xlsx", help="Path to PPB-Affinity Excel file.")
    parser.add_argument("--pdb-dir", default="data/pdb", help="Directory containing or receiving RCSB PDB files.")
    parser.add_argument("--out-dir", default="outputs/task1", help="Directory for Task 1 outputs.")
    parser.add_argument("--download", action="store_true", help="Download missing PDB files from RCSB.")
    parser.add_argument("--overwrite-downloads", action="store_true", help="Re-download PDB files even if present.")
    parser.add_argument(
        "--insecure-downloads",
        action="store_true",
        help="Disable HTTPS certificate verification for RCSB downloads if local CA certificates are broken.",
    )
    parser.add_argument("--pdb-id", action="append", dest="pdb_ids", help="Restrict processing to one PDB ID; repeatable.")
    parser.add_argument(
        "--dimers-only",
        action="store_true",
        help="Process only complexes with exactly one ligand chain and one receptor chain.",
    )
    parser.add_argument("--start", type=int, default=0, help="First dataset row index to process.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process.")
    parser.add_argument("--contact-cutoff", type=float, default=5.0, help="Heavy-atom interface cutoff in Angstrom.")
    parser.add_argument(
        "--surface-backend",
        choices=["sampler", "chimerax", "both"],
        default="chimerax",
        help=(
            "Surface generator. Default ChimeraX computes full ligand/receptor chain surfaces "
            "and then filters them to interface residues."
        ),
    )
    parser.add_argument(
        "--chimerax-full-surface-filter",
        dest="chimerax_full_surface_filter",
        action="store_true",
        default=True,
        help="Compute ChimeraX surfaces on full chains and filter to nearest interface residues.",
    )
    parser.add_argument(
        "--no-chimerax-full-surface-filter",
        dest="chimerax_full_surface_filter",
        action="store_false",
        help="Use legacy ChimeraX behavior on already-cut interface patch PDBs.",
    )
    parser.add_argument("--chimerax-bin", default=None, help="Path to ChimeraX executable.")
    parser.add_argument(
        "--chimerax-csv-stride",
        type=int,
        default=1,
        help="Keep every Nth ChimeraX OBJ vertex when converting to CSV.",
    )
    parser.add_argument("--surface-samples", type=int, default=64, help="Surface sample points per interface atom.")
    parser.add_argument("--probe-radius", type=float, default=1.4, help="Probe radius used for accessible surface sampling.")
    parser.add_argument("--density-alpha", type=float, default=1.8, help="Gaussian decay for approximate electron density.")
    parser.add_argument("--density-cutoff", type=float, default=6.0, help="Max Angstrom distance for density contributions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        dataset_path=Path(args.dataset),
        pdb_dir=Path(args.pdb_dir),
        output_dir=Path(args.out_dir),
        download_missing=args.download,
        overwrite_downloads=args.overwrite_downloads,
        download_verify_ssl=not args.insecure_downloads,
        contact_cutoff_angstrom=args.contact_cutoff,
        dimers_only=args.dimers_only,
        surface_backend=args.surface_backend,
        chimerax_bin=args.chimerax_bin,
        chimerax_csv_stride=args.chimerax_csv_stride,
        chimerax_full_surface_filter=args.chimerax_full_surface_filter,
        surface=SurfaceConfig(
            samples_per_atom=args.surface_samples,
            probe_radius=args.probe_radius,
            density_alpha=args.density_alpha,
            density_cutoff_angstrom=args.density_cutoff,
        ),
    )
    manifest = run_pipeline(config, pdb_ids=args.pdb_ids, start=args.start, limit=args.limit)
    ok = int((manifest["status"] == "ok").sum()) if "status" in manifest else 0
    errors = int((manifest["status"] == "error").sum()) if "status" in manifest else 0
    print(f"Task 1 completed: {ok} ok, {errors} errors. Manifest: {config.output_dir / 'manifest.csv'}")
    if errors:
        print(manifest.loc[manifest["status"] == "error", ["row_index", "pdb_id", "error"]].to_string(index=False))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
