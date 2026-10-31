"""Critical and ignored regions of an artwork.

* **Ignored** regions are not inspected: they are removed from the comparison
  mask (no structural defects, no colour measurement), text touching them is
  not read, and barcodes found there in the artwork are not inspected (barcodes
  in barcode zones still are). Use them for areas that legitimately differ from
  the artwork (variable data printed later, a window in the label, a varnish
  area that photographs differently).
* **Critical** regions are inspected more strictly: a structural defect or a
  colour finding touching one is at least ``decision.critical_region_severity``
  (major by default, so a reliable finding there fails the label), and an
  uncertain one always needs a person (REVIEW), whatever its size.

Text and barcodes have their own critical zones (core/text_zones.py,
core/barcode_zones.py). Regions are stored next to the artwork in
``<artwork file name>.regions.json`` or in an inspection profile;
coordinates are fractions (0-1) of the artwork width and height.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from print_inspector.core.defects import DefectReport, DefectSeverity

REGIONS_SUFFIX = ".regions.json"
FORMAT_VERSION = 1

Box = tuple[int, int, int, int]


class RegionError(ValueError):
    """The region file is unreadable or invalid."""


class RegionKind(str, Enum):
    CRITICAL = "critical"
    IGNORE = "ignore"

    @property
    def label(self) -> str:
        return {RegionKind.CRITICAL: "Critical", RegionKind.IGNORE: "Ignored"}[self]


@dataclass(frozen=True)
class InspectionRegion:
    name: str
    kind: RegionKind
    x: float
    y: float
    width: float
    height: float

    def rect(self, width: int, height: int) -> Box:
        """(x, y, w, h) in an image of ``width`` x ``height`` showing the whole artwork."""
        x0, y0 = round(self.x * width), round(self.y * height)
        x1, y1 = round((self.x + self.width) * width), round((self.y + self.height) * height)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def regions_path(artwork_path: Path) -> Path:
    artwork_path = Path(artwork_path)
    return artwork_path.with_name(artwork_path.name + REGIONS_SUFFIX)


def _validate(region: InspectionRegion, index: int) -> None:
    where = f"region {index + 1}"
    if not region.name.strip():
        raise RegionError(f"{where} has no name.")
    if not isinstance(region.kind, RegionKind):
        raise RegionError(f"{where} ('{region.name}') has an unknown type.")
    if not (0 <= region.x < 1 and 0 <= region.y < 1 and region.width > 0 and region.height > 0):
        raise RegionError(f"{where} ('{region.name}') lies outside the artwork.")
    if region.x + region.width > 1.0001 or region.y + region.height > 1.0001:
        raise RegionError(f"{where} ('{region.name}') extends beyond the artwork.")


def validate_regions(regions) -> None:
    """Raises RegionError for an invalid region or a repeated name."""
    names = set()
    for index, region in enumerate(regions):
        _validate(region, index)
        if region.name in names:
            raise RegionError(f"Region name '{region.name}' is used twice.")
        names.add(region.name)


def regions_from_data(items: Any, source: str) -> tuple[InspectionRegion, ...]:
    """Regions from a list of JSON objects (``source`` names the file for messages). Raises RegionError."""
    if not isinstance(items, list):
        raise RegionError(f"{source} has no list of regions.")
    regions = []
    for index, raw in enumerate(items):
        try:
            kind = RegionKind(str(raw["kind"]).lower())
            region = InspectionRegion(
                name=str(raw["name"]), kind=kind,
                x=float(raw["x"]), y=float(raw["y"]), width=float(raw["width"]), height=float(raw["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegionError(f"Region {index + 1} in {source} is incomplete or has an unknown type: {exc}") from exc
        regions.append(region)
    validate_regions(regions)
    return tuple(regions)


def regions_to_data(regions) -> list[dict[str, Any]]:
    return [{**dataclasses.asdict(r), "kind": r.kind.value} for r in regions]


def load_regions(path: Path) -> tuple[InspectionRegion, ...]:
    """Regions from ``path``; none if the file does not exist. Raises RegionError."""
    path = Path(path)
    if not path.is_file():
        return ()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegionError(f"Region file {path.name} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise RegionError(f"Region file {path.name} has no 'regions' list.")
    return regions_from_data(data.get("regions"), f"Region file {path.name}")


def save_regions(path: Path, regions) -> None:
    """Write the regions atomically; an empty list removes the file. Raises RegionError / OSError."""
    path = Path(path)
    validate_regions(regions)
    if not regions:
        if path.exists():
            path.unlink()
        return
    data = {"version": FORMAT_VERSION, "regions": regions_to_data(regions)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- use in the inspection


def boxes(regions, kind: RegionKind, width: int, height: int) -> tuple[Box, ...]:
    """Rectangles of the regions of ``kind`` in an image of ``width`` x ``height``."""
    return tuple(r.rect(width, height) for r in regions if r.kind is kind)


def inspection_mask(mask: np.ndarray, regions) -> np.ndarray:
    """``mask`` without the ignored regions (the same array if there are none)."""
    ignored = boxes(regions, RegionKind.IGNORE, mask.shape[1], mask.shape[0])
    if not ignored:
        return mask
    out = mask.copy()
    for x, y, w, h in ignored:
        out[y : y + h, x : x + w] = 0
    return out


def _overlaps(a: Box, b: Box) -> bool:
    return not (a[0] >= b[0] + b[2] or b[0] >= a[0] + a[2] or a[1] >= b[1] + b[3] or b[1] >= a[1] + a[3])


def critical_region_at(regions, box: Box, width: int, height: int) -> InspectionRegion | None:
    """The first critical region that ``box`` (grid pixels of a ``width`` x ``height`` grid) touches."""
    return next(
        (r for r in regions if r.kind is RegionKind.CRITICAL and _overlaps(box, r.rect(width, height))), None
    )


def _raise(severity: DefectSeverity, minimum: DefectSeverity) -> DefectSeverity:
    return minimum if minimum.rank > severity.rank else severity


def apply_to_defects(report: DefectReport, regions, grid_shape: tuple[int, int], min_severity: str) -> DefectReport:
    """Mark defects touching critical regions and raise their severity to at least ``min_severity``."""
    if not any(r.kind is RegionKind.CRITICAL for r in regions):
        return report
    height, width = grid_shape[:2]
    minimum = DefectSeverity(min_severity)
    defects = []
    for d in report.defects:
        region = critical_region_at(regions, (d.x, d.y, d.width, d.height), width, height)
        if region is None:
            defects.append(d)
            continue
        severity = _raise(d.severity, minimum)
        note = f"In critical region '{region.name}'" + (
            f": severity raised from {d.severity.value} to {severity.value}." if severity is not d.severity else "."
        )
        defects.append(dataclasses.replace(d, severity=severity, severity_notes=d.severity_notes + (note,),
                                           critical_region=region.name))
    defects.sort(key=lambda d: (-d.severity.rank, -d.score))
    return dataclasses.replace(report, defects=tuple(defects))


def apply_to_color(report, regions, grid_shape: tuple[int, int], min_severity: str):
    """The same for the findings of a ColorReport (``critical`` is set on those in critical regions)."""
    if report is None or not any(r.kind is RegionKind.CRITICAL for r in regions):
        return report
    height, width = grid_shape[:2]
    minimum = DefectSeverity(min_severity)
    findings = []
    for f in report.findings:
        region = critical_region_at(regions, (f.x, f.y, f.width, f.height), width, height)
        if region is None:
            findings.append(f)
            continue
        severity = _raise(f.severity, minimum)
        note = f"In critical region '{region.name}'" + (
            f": severity raised from {f.severity.value} to {severity.value}." if severity is not f.severity else "."
        )
        findings.append(dataclasses.replace(f, severity=severity, critical=True, notes=f.notes + (note,)))
    return dataclasses.replace(report, findings=tuple(findings))
