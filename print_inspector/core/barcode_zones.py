"""Barcode inspection zones of an artwork.

A zone marks where a barcode must be and, optionally, its format and exact
value (for example the GTIN from the product master data). Zones are stored
next to the artwork in ``<artwork file name>.barcodes.json``; coordinates are
fractions (0-1) of the artwork width and height.

Without zones, every supported barcode found in the artwork is inspected and
compared with the value decoded from the artwork.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from print_inspector.config import BARCODE_FORMATS

ZONES_SUFFIX = ".barcodes.json"
FORMAT_VERSION = 1
ANY_FORMAT = "any"


class BarcodeZoneError(ValueError):
    """The zone file is unreadable or invalid."""


@dataclass(frozen=True)
class BarcodeZone:
    name: str
    x: float
    y: float
    width: float
    height: float
    format: str = ANY_FORMAT  # "any" or one of BARCODE_FORMATS
    expected_value: str = ""  # empty: compare with the value decoded from the artwork
    critical: bool = True

    def rect(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0, y0 = round(self.x * width), round(self.y * height)
        x1, y1 = round((self.x + self.width) * width), round((self.y + self.height) * height)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def zones_path(artwork_path: Path) -> Path:
    artwork_path = Path(artwork_path)
    return artwork_path.with_name(artwork_path.name + ZONES_SUFFIX)


def _validate(zone: BarcodeZone, index: int) -> None:
    where = f"barcode zone {index + 1}"
    if not zone.name.strip():
        raise BarcodeZoneError(f"{where} has no name.")
    if zone.format != ANY_FORMAT and zone.format not in BARCODE_FORMATS:
        raise BarcodeZoneError(f"{where} ('{zone.name}') has an unsupported format '{zone.format}'.")
    if not (0 <= zone.x < 1 and 0 <= zone.y < 1 and zone.width > 0 and zone.height > 0):
        raise BarcodeZoneError(f"{where} ('{zone.name}') lies outside the artwork.")
    if zone.x + zone.width > 1.0001 or zone.y + zone.height > 1.0001:
        raise BarcodeZoneError(f"{where} ('{zone.name}') extends beyond the artwork.")
    if zone.expected_value and zone.format in ("ean13", "upca"):
        from print_inspector.core.barcodes import gtin_check_digit_ok

        digits = "".join(c for c in zone.expected_value if c.isdigit())
        if len(digits) not in (12, 13) or len(digits) != len(zone.expected_value.strip()) or not gtin_check_digit_ok(digits):
            raise BarcodeZoneError(
                f"{where} ('{zone.name}'): '{zone.expected_value}' is not a valid EAN-13 / UPC-A "
                "(12 or 13 digits with a correct check digit)."
            )


def zones_from_data(items, source: str) -> tuple[BarcodeZone, ...]:
    """Zones from a list of JSON objects (``source`` names the file for messages). Raises BarcodeZoneError."""
    if not isinstance(items, list):
        raise BarcodeZoneError(f"{source} has no 'zones' list.")
    zones = []
    for index, raw in enumerate(items):
        try:
            zone = BarcodeZone(
                name=str(raw["name"]),
                x=float(raw["x"]), y=float(raw["y"]), width=float(raw["width"]), height=float(raw["height"]),
                format=str(raw.get("format", ANY_FORMAT)).lower(),
                expected_value=str(raw.get("expected_value", "")).strip(),
                critical=bool(raw.get("critical", True)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BarcodeZoneError(f"Barcode zone {index + 1} in {source} is incomplete: {exc}") from exc
        _validate(zone, index)
        zones.append(zone)
    return tuple(zones)


def zones_to_data(zones) -> list[dict]:
    return [asdict(z) for z in zones]


def load_zones(path: Path) -> tuple[BarcodeZone, ...]:
    """Zones from ``path``; none if the file does not exist. Raises BarcodeZoneError."""
    path = Path(path)
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BarcodeZoneError(f"Barcode zone file {path.name} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise BarcodeZoneError(f"Barcode zone file {path.name} has no 'zones' list.")
    return zones_from_data(data.get("zones"), f"Barcode zone file {path.name}")


def validate_zones(zones) -> None:
    for index, zone in enumerate(zones):
        _validate(zone, index)


def save_zones(path: Path, zones) -> None:
    """Write the zones atomically; an empty list removes the file. Raises BarcodeZoneError / OSError."""
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
