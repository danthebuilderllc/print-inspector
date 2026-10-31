"""Critical text zones of an artwork.

Zones mark text that must be checked strictly (lot numbers, dates, doses,
prices ...). They are stored next to the artwork in
``<artwork file name>.zones.json`` so they apply to every inspection of that
artwork. Coordinates are fractions (0-1) of the artwork width and height, so
they do not depend on the resolution of any image.

A zone may give the exact ``expected_text``; the print is then compared with
that text instead of with the text read from the artwork.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ZONES_SUFFIX = ".zones.json"
FORMAT_VERSION = 1


class TextZoneError(ValueError):
    """The zone file is unreadable or invalid."""


@dataclass(frozen=True)
class TextZone:
    name: str
    x: float
    y: float
    width: float
    height: float
    expected_text: str = ""
    critical: bool = True

    def rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) in an image of ``width`` x ``height`` showing the whole artwork."""
        x0, y0 = round(self.x * width), round(self.y * height)
        x1, y1 = round((self.x + self.width) * width), round((self.y + self.height) * height)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def contains(self, px: float, py: float, width: int, height: int) -> bool:
        x, y, w, h = self.rect(width, height)
        return x <= px < x + w and y <= py < y + h


def zones_path(artwork_path: Path) -> Path:
    artwork_path = Path(artwork_path)
    return artwork_path.with_name(artwork_path.name + ZONES_SUFFIX)


def _validate(zone: TextZone, index: int) -> None:
    where = f"zone {index + 1}"
    if not zone.name.strip():
        raise TextZoneError(f"{where} has no name.")
    if not (0 <= zone.x < 1 and 0 <= zone.y < 1 and zone.width > 0 and zone.height > 0):
        raise TextZoneError(f"{where} ('{zone.name}') lies outside the artwork.")
    if zone.x + zone.width > 1.0001 or zone.y + zone.height > 1.0001:
        raise TextZoneError(f"{where} ('{zone.name}') extends beyond the artwork.")


def zones_from_data(items, source: str) -> tuple[TextZone, ...]:
    """Zones from a list of JSON objects (``source`` names the file for messages). Raises TextZoneError."""
    if not isinstance(items, list):
        raise TextZoneError(f"{source} has no 'zones' list.")
    zones = []
    for index, raw in enumerate(items):
        try:
            zone = TextZone(
                name=str(raw["name"]),
                x=float(raw["x"]), y=float(raw["y"]),
                width=float(raw["width"]), height=float(raw["height"]),
                expected_text=str(raw.get("expected_text", "")),
                critical=bool(raw.get("critical", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TextZoneError(f"Text zone {index + 1} in {source} is incomplete: {exc}") from exc
        _validate(zone, index)
        zones.append(zone)
    return tuple(zones)


def zones_to_data(zones) -> list[dict]:
    return [asdict(z) for z in zones]


def validate_zones(zones) -> None:
    for index, zone in enumerate(zones):
        _validate(zone, index)


def load_zones(path: Path) -> tuple[TextZone, ...]:
    """Zones from ``path``; none if the file does not exist. Raises TextZoneError."""
    path = Path(path)
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextZoneError(f"Text zone file {path.name} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise TextZoneError(f"Text zone file {path.name} has no 'zones' list.")
    return zones_from_data(data.get("zones"), f"Text zone file {path.name}")


def save_zones(path: Path, zones: tuple[TextZone, ...] | list[TextZone]) -> None:
    """Write the zones (atomically); an empty list removes the file. Raises TextZoneError / OSError."""
    path = Path(path)
    validate_zones(zones)
    if not zones:
        if path.exists():
            path.unlink()
        return
    data = {"version": FORMAT_VERSION, "zones": zones_to_data(zones)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
