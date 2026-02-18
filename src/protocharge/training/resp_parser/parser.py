from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Frame:
    center_of_mass: tuple[float, float, float] | None = None
    dipole_moment_vector: tuple[float, float, float] | None = None
    dipole_moment_magnitude: float | None = None
    positions: List[Tuple[float, float, float]] = field(default_factory=list)
    esp_charges: List[float] = field(default_factory=list)
    exposure_fractions: List[float] = field(default_factory=list)
    esp_rms_error: float | None = None
    resp_charges: List[float] = field(default_factory=list)
    resp_rms_error: float | None = None

@dataclass
class ESPGridFrame:
    coordinates: List[Tuple[float, float, float]]
    potentials: List[float]


@dataclass
class Elements:
    symbols: List[str]
    coordinates: List[Tuple[float, float, float]]

class ParseRespDotOut:
    def __init__(self, filename: Path | str, number_of_atoms: int) -> None:
        self.file = Path(filename)
        self.number_of_atoms = number_of_atoms

    def success_check(self) -> bool:
        success_string = "| Job finished:"
        with self.file.open() as f:
            for line in f:
                if line.strip().startswith(success_string):
                    return True
        return False

    def extract_frames(self) -> List[Frame]:
        lines = self._read_lines()

        def _parse_braced_triplet(text: str) -> tuple[float, float, float]:
            start = text.find("{")
            end = text.find("}", start + 1)
            if start == -1 or end == -1:
                raise ValueError(f"Could not find braced triplet in line: {text!r}")
            values = text[start + 1 : end].replace(",", " ").split()
            if len(values) != 3:
                raise ValueError(f"Expected 3 values inside braces, got {values!r}")
            return tuple(float(v) for v in values)

        def _parse_rrms(text: str) -> float:
            if ":" not in text:
                raise ValueError(f"Malformed RRMS line: {text!r}")
            return float(text.split(":", 1)[1].strip())

        def _parse_dipole(text: str) -> tuple[tuple[float, float, float], float | None]:
            vec = _parse_braced_triplet(text)
            magnitude: float | None = None
            marker = "(|D|"
            start = text.find(marker)
            if start != -1:
                sub = text[start:]
                eq_idx = sub.find("=")
                close_idx = sub.find(")", eq_idx)
                if eq_idx != -1 and close_idx != -1:
                    magnitude = float(sub[eq_idx + 1 : close_idx].strip())
            return vec, magnitude

        def _is_separator(text: str) -> bool:
            stripped = text.strip()
            return bool(stripped) and set(stripped) == {"-"}

        frames: List[Frame] = []
        current_frame: Frame | None = None

        idx = 0
        total = len(lines)
        while idx < total:
            line = lines[idx]
            stripped = line.strip()

            if stripped.startswith("CENTER OF MASS:"):
                current_frame = Frame()
                current_frame.center_of_mass = _parse_braced_triplet(line)
                frames.append(current_frame)
                idx += 1
                continue

            if stripped.startswith("DIPOLE MOMENT:"):
                if current_frame is None:
                    current_frame = Frame()
                    frames.append(current_frame)
                current_frame.dipole_moment_vector, current_frame.dipole_moment_magnitude = _parse_dipole(line)
                idx += 1
                continue

            if stripped == "ESP unrestrained charges:":
                if current_frame is None:
                    current_frame = Frame()
                    frames.append(current_frame)
                idx += 1  # move to column header
                if idx < total and lines[idx].strip().startswith("Atom"):
                    idx += 1
                if idx < total and _is_separator(lines[idx]):
                    idx += 1

                positions: List[Tuple[float, float, float]] = []
                esp_charges: List[float] = []
                exposures: List[float] = []
                for _ in range(self.number_of_atoms):
                    if idx >= total:
                        raise ValueError("Unexpected end of file while reading ESP unrestrained charges block")
                    parts = lines[idx].split()
                    if len(parts) < 6:
                        raise ValueError(f"Malformed ESP unrestrained charge line: {lines[idx]!r}")
                    positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    esp_charges.append(float(parts[4]))
                    exposures.append(float(parts[5]))
                    idx += 1

                if idx < total and _is_separator(lines[idx]):
                    idx += 1
                while idx < total and not lines[idx].strip():
                    idx += 1

                esp_rms: float | None = None
                if idx < total and lines[idx].strip().startswith("Quality of fit"):
                    esp_rms = _parse_rrms(lines[idx])
                    idx += 1

                current_frame.positions = positions
                current_frame.esp_charges = esp_charges
                current_frame.exposure_fractions = exposures
                current_frame.esp_rms_error = esp_rms
                continue

            if stripped == "ESP restrained charges:":
                if current_frame is None:
                    current_frame = Frame()
                    frames.append(current_frame)
                idx += 1  # move to column header
                if idx < total and lines[idx].strip().startswith("Atom"):
                    idx += 1
                if idx < total and _is_separator(lines[idx]):
                    idx += 1

                resp_charges: List[float] = []
                for _ in range(self.number_of_atoms):
                    if idx >= total:
                        raise ValueError("Unexpected end of file while reading ESP restrained charges block")
                    parts = lines[idx].split()
                    if len(parts) < 5:
                        raise ValueError(f"Malformed ESP restrained charge line: {lines[idx]!r}")
                    resp_charges.append(float(parts[4]))
                    idx += 1

                if idx < total and _is_separator(lines[idx]):
                    idx += 1
                while idx < total and not lines[idx].strip():
                    idx += 1

                resp_rms: float | None = None
                if idx < total and lines[idx].strip().startswith("Quality of fit"):
                    resp_rms = _parse_rrms(lines[idx])
                    idx += 1

                current_frame.resp_charges = resp_charges
                current_frame.resp_rms_error = resp_rms
                continue

            idx += 1

        return frames

    def _read_lines(self) -> List[str]:
        with self.file.open() as f:
            return [line.rstrip("\n") for line in f]


def _iter_xyz_blocks(file_path: Path) -> List[Tuple[int, str, List[str]]]:
    with file_path.open() as fh:
        lines = [line.rstrip("\n") for line in fh]

    idx = 0
    total = len(lines)
    blocks: List[Tuple[int, str, List[str]]] = []

    while idx < total:
        while idx < total and not lines[idx].strip():
            idx += 1
        if idx >= total:
            break

        header = lines[idx].strip()
        try:
            natoms = int(header)
        except ValueError as exc:
            raise ValueError(f"Expected atom count at line {idx + 1}, got {header!r}") from exc
        idx += 1

        comment = lines[idx] if idx < total else ""
        idx += 1

        rows: List[str] = []
        count = 0
        while idx < total and count < natoms:
            line = lines[idx]
            if not line.strip():
                idx += 1
                continue
            rows.append(line)
            count += 1
            idx += 1

        if count != natoms:
            raise ValueError(f"xyz block declared {natoms} atoms but found {count}")

        blocks.append((natoms, comment, rows))

    return blocks


class ParseESPXYZ:
    def __init__(self, filename: Path | str) -> None:
        self.file = Path(filename)

    def frames(self) -> List[ESPGridFrame]:
        frames: List[ESPGridFrame] = []
        for _natoms, _comment, rows in _iter_xyz_blocks(self.file):
            coords: List[Tuple[float, float, float]] = []
            potentials: List[float] = []
            for row in rows:
                parts = row.split()
                if len(parts) < 5:
                    raise ValueError(f"Malformed esp.xyz line: {row!r}")
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
                potentials.append(float(parts[4]))
            frames.append(ESPGridFrame(coords, potentials))
        return frames


class ParseDotXYZ:
    def __init__(self, filename: Path | str) -> None:
        self.file = Path(filename)

    def elements(self) -> List[Elements]:
        frames: List[Elements] = []
        for natoms, _comment, rows in _iter_xyz_blocks(self.file):
            symbols: List[str] = []
            coords: List[Tuple[float, float, float]] = []
            for row in rows:
                parts = row.split()
                if len(parts) < 4:
                    raise ValueError("Encountered malformed atom line in xyz block")
                symbols.append(parts[0])
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if len(symbols) != natoms:
                raise ValueError("Mismatch between declared atoms and parsed elements in xyz block")
            frames.append(Elements(symbols=symbols, coordinates=coords))
        return frames
