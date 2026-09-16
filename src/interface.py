from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pdb import Atom, coords_array


@dataclass(frozen=True)
class InterfaceContact:
    ligand_chain: str
    ligand_residue_number: int
    ligand_insertion_code: str
    ligand_residue_name: str
    receptor_chain: str
    receptor_residue_number: int
    receptor_insertion_code: str
    receptor_residue_name: str
    min_distance_angstrom: float
    atom_contact_count: int


@dataclass(frozen=True)
class InterfaceResult:
    ligand_residues: frozenset[tuple[str, int, str, str]]
    receptor_residues: frozenset[tuple[str, int, str, str]]
    contacts: tuple[InterfaceContact, ...]


def _contact_from_keys(
    ligand_key: tuple[str, int, str, str],
    receptor_key: tuple[str, int, str, str],
    min_distance: float,
    atom_contact_count: int,
) -> InterfaceContact:
    return InterfaceContact(
        ligand_chain=ligand_key[0],
        ligand_residue_number=ligand_key[1],
        ligand_insertion_code=ligand_key[2],
        ligand_residue_name=ligand_key[3],
        receptor_chain=receptor_key[0],
        receptor_residue_number=receptor_key[1],
        receptor_insertion_code=receptor_key[2],
        receptor_residue_name=receptor_key[3],
        min_distance_angstrom=float(min_distance),
        atom_contact_count=int(atom_contact_count),
    )


def find_interface(
    ligand_atoms: list[Atom],
    receptor_atoms: list[Atom],
    *,
    cutoff_angstrom: float = 5.0,
    chunk_size: int = 512,
) -> InterfaceResult:
    """Find contacting residues using heavy-atom distances across both partners."""

    if not ligand_atoms or not receptor_atoms:
        return InterfaceResult(frozenset(), frozenset(), tuple())

    ligand_coords = coords_array(ligand_atoms)
    receptor_coords = coords_array(receptor_atoms)
    cutoff_sq = cutoff_angstrom * cutoff_angstrom

    ligand_residues: set[tuple[str, int, str, str]] = set()
    receptor_residues: set[tuple[str, int, str, str]] = set()
    residue_pair_stats: dict[
        tuple[tuple[str, int, str, str], tuple[str, int, str, str]], list[float | int]
    ] = {}

    for start in range(0, len(ligand_atoms), chunk_size):
        stop = min(start + chunk_size, len(ligand_atoms))
        delta = ligand_coords[start:stop, None, :] - receptor_coords[None, :, :]
        distances_sq = np.einsum("ijk,ijk->ij", delta, delta)
        rows, cols = np.where(distances_sq <= cutoff_sq)
        if len(rows) == 0:
            continue

        for local_row, receptor_idx in zip(rows, cols):
            ligand_idx = start + int(local_row)
            receptor_idx = int(receptor_idx)
            ligand_key = ligand_atoms[ligand_idx].residue_key
            receptor_key = receptor_atoms[receptor_idx].residue_key
            ligand_residues.add(ligand_key)
            receptor_residues.add(receptor_key)

            pair_key = (ligand_key, receptor_key)
            distance = float(np.sqrt(distances_sq[local_row, receptor_idx]))
            current = residue_pair_stats.get(pair_key)
            if current is None:
                residue_pair_stats[pair_key] = [distance, 1]
            else:
                current[0] = min(float(current[0]), distance)
                current[1] = int(current[1]) + 1

    contacts = tuple(
        sorted(
            (
                _contact_from_keys(lig_key, rec_key, float(stats[0]), int(stats[1]))
                for (lig_key, rec_key), stats in residue_pair_stats.items()
            ),
            key=lambda c: (
                c.min_distance_angstrom,
                c.ligand_chain,
                c.ligand_residue_number,
                c.receptor_chain,
                c.receptor_residue_number,
            ),
        )
    )

    return InterfaceResult(frozenset(ligand_residues), frozenset(receptor_residues), contacts)


def atoms_in_residues(atoms: list[Atom], residues: frozenset[tuple[str, int, str, str]]) -> list[Atom]:
    return [atom for atom in atoms if atom.residue_key in residues]
