from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ComplexRecord:
    """One PPB-Affinity complex row needed for Task 1."""

    row_index: int
    pdb_id: str
    ligand_chains: tuple[str, ...]
    receptor_chains: tuple[str, ...]
    kd_m: float | None
    source_dataset: str | None = None
    ligand_name: str | None = None
    receptor_name: str | None = None
    mutations: str | None = None
    model: str | None = None


def _as_optional_str(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def parse_chain_list(value: object) -> tuple[str, ...]:
    """Parse chain lists from PPB-Affinity cells such as ``H,L`` or ``A;B``."""

    text = _as_optional_str(value)
    if not text:
        return tuple()
    parts = [p.strip() for p in re.split(r"[,;/|\s]+", text) if p.strip()]
    return tuple(dict.fromkeys(parts))


def _as_optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_complex_records(
    dataset_path: str | Path,
    *,
    pdb_ids: Iterable[str] | None = None,
    start: int = 0,
    limit: int | None = None,
    dimers_only: bool = False,
) -> list[ComplexRecord]:
    """Read PPB-Affinity records from the local Excel workbook."""

    path = Path(dataset_path)
    df = pd.read_excel(path)

    required = {"PDB", "Ligand Chains", "Receptor Chains"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} is missing required columns: {sorted(missing)}")

    allowed = None if pdb_ids is None else {p.upper() for p in pdb_ids}
    records: list[ComplexRecord] = []

    for row_index, row in df.iterrows():
        pdb_id = _as_optional_str(row.get("PDB"))
        if not pdb_id:
            continue
        pdb_id = pdb_id.upper()
        if allowed is not None and pdb_id not in allowed:
            continue
        if row_index < start:
            continue

        ligand_chains = parse_chain_list(row.get("Ligand Chains"))
        receptor_chains = parse_chain_list(row.get("Receptor Chains"))
        if not ligand_chains or not receptor_chains:
            continue
        if dimers_only and (len(ligand_chains) != 1 or len(receptor_chains) != 1):
            continue

        records.append(
            ComplexRecord(
                row_index=int(row_index),
                pdb_id=pdb_id,
                ligand_chains=ligand_chains,
                receptor_chains=receptor_chains,
                kd_m=_as_optional_float(row.get("KD(M)")),
                source_dataset=_as_optional_str(row.get("Source Data Set")),
                ligand_name=_as_optional_str(row.get("Ligand Name")),
                receptor_name=_as_optional_str(row.get("Receptor Name")),
                mutations=_as_optional_str(row.get("Mutations")),
                model=_as_optional_str(row.get("Model")),
            )
        )
        if limit is not None and len(records) >= limit:
            break

    return records
