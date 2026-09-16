from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ssl
import urllib.request

import numpy as np


WATER_RESIDUES = {"HOH", "WAT", "DOD"}


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    x: float
    y: float
    z: float
    element: str
    record_name: str = "ATOM"

    @property
    def coord(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z), dtype=float)

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code.strip(),
            self.residue_name,
        )


def _infer_element(atom_name: str, element_field: str) -> str:
    element = element_field.strip().upper()
    if element:
        return element
    stripped = atom_name.strip()
    stripped = stripped[1:] if stripped and stripped[0].isdigit() else stripped
    return (stripped[:2] if len(stripped) >= 2 and stripped[:2].title() in {"Cl", "Br"} else stripped[:1]).upper()


def is_hydrogen(atom: Atom) -> bool:
    return atom.element.upper() == "H" or atom.name.strip().upper().startswith("H")


def read_pdb(path: str | Path, *, include_hetatm: bool = True, keep_hydrogen: bool = False) -> list[Atom]:
    """Parse a PDB file using only the fixed-width ATOM/HETATM records."""

    atoms: list[Atom] = []
    allowed_records = {"ATOM", "HETATM"} if include_hetatm else {"ATOM"}

    with Path(path).open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record_name = line[0:6].strip()
            if record_name not in allowed_records:
                continue
            if len(line) < 54:
                continue

            altloc = line[16].strip()
            if altloc not in {"", "A", "1"}:
                continue

            residue_name = line[17:20].strip()
            if residue_name in WATER_RESIDUES:
                continue

            try:
                atom = Atom(
                    serial=int(line[6:11]),
                    name=line[12:16].strip(),
                    residue_name=residue_name,
                    chain_id=line[21].strip() or "_",
                    residue_number=int(line[22:26]),
                    insertion_code=line[26].strip(),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    element=_infer_element(line[12:16], line[76:78] if len(line) >= 78 else ""),
                    record_name=record_name,
                )
            except ValueError:
                continue

            if not keep_hydrogen and is_hydrogen(atom):
                continue
            atoms.append(atom)

    return atoms


def format_pdb_atom(atom: Atom, serial: int | None = None) -> str:
    """Format an atom as a standard PDB ATOM/HETATM line."""

    atom_serial = atom.serial if serial is None else serial
    insertion_code = atom.insertion_code[:1] if atom.insertion_code else " "
    chain_id = atom.chain_id[:1] if atom.chain_id else " "
    element = atom.element.strip().upper()[:2].rjust(2)
    return (
        f"{atom.record_name:<6}{atom_serial:5d} "
        f"{atom.name:>4} "
        f"{atom.residue_name:>3} {chain_id}"
        f"{atom.residue_number:4d}{insertion_code}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {element}"
    )


def write_pdb(path: str | Path, atoms: list[Atom], *, renumber: bool = True) -> None:
    """Write atoms to a small PDB file for downstream surface tools."""

    lines = []
    for idx, atom in enumerate(atoms, start=1):
        lines.append(format_pdb_atom(atom, serial=idx if renumber else None))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _download_ssl_context(*, verify_ssl: bool) -> ssl.SSLContext:
    if not verify_ssl:
        return ssl._create_unverified_context()

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download_pdb(
    pdb_id: str,
    pdb_dir: str | Path,
    *,
    overwrite: bool = False,
    verify_ssl: bool = True,
) -> Path:
    """Download one PDB file from RCSB into ``pdb_dir``."""

    pdb_dir = Path(pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    target = pdb_dir / f"{pdb_id}.pdb"
    if target.exists() and not overwrite:
        return target

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    request = urllib.request.Request(url, headers={"User-Agent": "aml-task1-interface-pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=45, context=_download_ssl_context(verify_ssl=verify_ssl)) as response:
        data = response.read()
    if not data.startswith((b"HEADER", b"TITLE", b"ATOM", b"MODEL")):
        raise RuntimeError(f"Unexpected response while downloading {pdb_id} from {url}")
    target.write_bytes(data)
    return target


def select_chains(atoms: list[Atom], chains: tuple[str, ...]) -> list[Atom]:
    wanted = set(chains)
    return [atom for atom in atoms if atom.chain_id in wanted]


def coords_array(atoms: list[Atom]) -> np.ndarray:
    if not atoms:
        return np.empty((0, 3), dtype=float)
    return np.array([(a.x, a.y, a.z) for a in atoms], dtype=float)
