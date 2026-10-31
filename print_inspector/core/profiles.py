"""Inspection profiles: reusable settings for a recurring label design.

A profile stores everything that belongs to a label design rather than to a
computer or a camera:

* the comparison sensitivity and the sensitivity thresholds (``comparison``,
  ``defects`` and ``decision`` settings, i.e. the acceptance criteria);
* critical and ignored regions (core/regions.py);
* text verification settings (``text``) and the text zones with their
  expected text (core/text_zones.py);
* barcode settings (``barcodes``) and the barcode zones with their expected
  values (core/barcode_zones.py);
* colour tolerances (``color``).

Paths that only make sense on one computer (Tesseract folder, tessdata
folder, colour calibration file) are not stored; they stay as configured on
the computer that loads the profile. Capture, alignment, registration, UI and
logging settings are not part of a profile either.

Profiles are JSON files in ``<user data dir>/profiles`` (one per profile).
They can be exported to and imported from any local file, e.g. a network
share or a USB stick; nothing is uploaded and no database is involved.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from print_inspector.config import (
    SENSITIVITY_LEVELS,
    AppConfig,
    ConfigError,
    config_data,
    deep_merge,
    default_config_data,
    parse_config,
    user_data_dir,
)
from print_inspector.core.barcode_zones import BarcodeZone, BarcodeZoneError
from print_inspector.core.barcode_zones import zones_from_data as barcode_zones_from_data
from print_inspector.core.barcode_zones import zones_to_data as barcode_zones_to_data
from print_inspector.core.regions import InspectionRegion, RegionError, RegionKind, regions_from_data, regions_to_data
from print_inspector.core.text_zones import TextZone, TextZoneError, zones_from_data, zones_to_data

FILE_FORMAT = "print-inspector-profile"
FORMAT_VERSION = 1
PROFILE_SUFFIX = ".json"
PROFILE_SECTIONS = ("comparison", "defects", "decision", "text", "barcodes", "color")
# Settings that describe this computer, not the label: never stored, never applied.
MACHINE_SETTINGS: dict[str, tuple[str, ...]] = {
    "text": ("tesseract_path", "tessdata_dir"),
    "color": ("calibration_profile",),
}
# Relative difference of the artwork aspect ratio above which the zones may not fit.
ASPECT_TOLERANCE = 0.02
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


class ProfileError(ValueError):
    """The profile is unreadable or invalid."""


@dataclass(frozen=True)
class InspectionProfile:
    name: str
    description: str
    sensitivity: str
    settings: Mapping[str, Mapping[str, Any]]  # section -> values, sections of PROFILE_SECTIONS
    regions: tuple[InspectionRegion, ...] = ()
    text_zones: tuple[TextZone, ...] = ()
    barcode_zones: tuple[BarcodeZone, ...] = ()
    artwork_name: str = ""  # file name of the artwork it was made with (information only)
    artwork_size: tuple[int, int] | None = None  # width, height in pixels
    created: str = ""  # ISO date-time
    modified: str = ""
    app_version: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)  # unknown top-level keys, kept

    def summary(self) -> str:
        critical = sum(r.kind is RegionKind.CRITICAL for r in self.regions)
        ignored = len(self.regions) - critical
        parts = [
            f"sensitivity {self.sensitivity}",
            f"{critical} critical / {ignored} ignored region(s)",
            f"{len(self.text_zones)} text zone(s)",
            f"{len(self.barcode_zones)} barcode zone(s)",
        ]
        return ", ".join(parts)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _label_settings(config: AppConfig) -> dict[str, dict[str, Any]]:
    data = config_data(config)
    return {
        section: {k: v for k, v in data[section].items() if k not in MACHINE_SETTINGS.get(section, ())}
        for section in PROFILE_SECTIONS
    }


def create_profile(
    name: str,
    description: str,
    config: AppConfig,
    sensitivity: str,
    regions=(),
    text_zones=(),
    barcode_zones=(),
    artwork_name: str = "",
    artwork_size: tuple[int, int] | None = None,
    previous: InspectionProfile | None = None,
) -> InspectionProfile:
    """A profile of the current inspection setup (``previous``: the profile it replaces)."""
    name = name.strip()
    if not name:
        raise ProfileError("The profile needs a name.")
    if sensitivity not in SENSITIVITY_LEVELS:
        raise ProfileError(f"Unknown sensitivity '{sensitivity}'.")
    now = _now()
    return InspectionProfile(
        name=name,
        description=description.strip(),
        sensitivity=sensitivity,
        settings=_label_settings(config),
        regions=tuple(regions),
        text_zones=tuple(text_zones),
        barcode_zones=tuple(barcode_zones),
        artwork_name=artwork_name,
        artwork_size=tuple(artwork_size) if artwork_size else None,
        created=previous.created if previous is not None and previous.created else now,
        modified=now,
        app_version=config.app.version,
        extra=dict(previous.extra) if previous is not None else {},
    )


def _merged(base_data: dict[str, Any], profile: InspectionProfile) -> dict[str, Any]:
    data = dict(base_data)
    for section, values in profile.settings.items():
        machine = {k: base_data[section][k] for k in MACHINE_SETTINGS.get(section, ()) if k in base_data[section]}
        data[section] = {**deep_merge(base_data[section], values), **machine}
    data["inspection"] = {**base_data["inspection"], "sensitivity": profile.sensitivity}
    return data


def apply_profile(profile: InspectionProfile, base: AppConfig) -> AppConfig:
    """``base`` with the profile's settings (machine paths and other sections stay). Raises ProfileError."""
    try:
        return parse_config(_merged(config_data(base), profile))
    except ConfigError as exc:
        raise ProfileError(f"Profile '{profile.name}' has invalid settings: {exc}") from exc


def is_profile_setting(section: str, key: str) -> bool:
    """True if a profile stores (and overrides) this setting; its sensitivity replaces the default one."""
    if (section, key) == ("inspection", "sensitivity"):
        return True
    return section in PROFILE_SECTIONS and key not in MACHINE_SETTINGS.get(section, ())


def matches(
    profile: InspectionProfile, config: AppConfig, sensitivity: str, regions, text_zones, barcode_zones
) -> bool:
    """True if the current setup is exactly what the profile stores (nothing changed since loading/saving)."""
    if (profile.sensitivity, tuple(profile.regions), tuple(profile.text_zones), tuple(profile.barcode_zones)) != (
        sensitivity, tuple(regions), tuple(text_zones), tuple(barcode_zones)
    ):
        return False
    try:
        return _label_settings(apply_profile(profile, config)) == _label_settings(config)
    except ProfileError:
        return False


def profile_to_data(profile: InspectionProfile) -> dict[str, Any]:
    return {
        **dict(profile.extra),
        "format": FILE_FORMAT,
        "version": FORMAT_VERSION,
        "name": profile.name,
        "description": profile.description,
        "artwork": {
            "file_name": profile.artwork_name,
            "width": profile.artwork_size[0] if profile.artwork_size else None,
            "height": profile.artwork_size[1] if profile.artwork_size else None,
        },
        "created": profile.created,
        "modified": profile.modified,
        "application_version": profile.app_version,
        "sensitivity": profile.sensitivity,
        "settings": {section: dict(values) for section, values in profile.settings.items()},
        "regions": regions_to_data(profile.regions),
        "text_zones": zones_to_data(profile.text_zones),
        "barcode_zones": barcode_zones_to_data(profile.barcode_zones),
    }


_KNOWN_KEYS = {
    "format", "version", "name", "description", "artwork", "created", "modified", "application_version",
    "sensitivity", "settings", "regions", "text_zones", "barcode_zones",
}


def profile_from_data(data: Any, source: str = "The profile") -> InspectionProfile:
    """Validate a profile mapping (settings, regions and zones). Raises ProfileError."""
    if not isinstance(data, dict) or data.get("format") != FILE_FORMAT:
        raise ProfileError(f"{source} is not a Print Inspector profile.")
    version = data.get("version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise ProfileError(f"{source} was made by a newer version of Print Inspector (format {version}).")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProfileError(f"{source} has no name.")
    sensitivity = str(data.get("sensitivity", "")).lower()
    if sensitivity not in SENSITIVITY_LEVELS:
        raise ProfileError(f"{source}: the sensitivity must be one of {', '.join(SENSITIVITY_LEVELS)}.")
    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ProfileError(f"{source}: 'settings' must be an object.")
    unknown = sorted(set(settings) - set(PROFILE_SECTIONS))
    if unknown:
        raise ProfileError(f"{source} contains settings that a profile cannot change: {', '.join(unknown)}.")
    for section, values in settings.items():
        if not isinstance(values, dict):
            raise ProfileError(f"{source}: settings section '{section}' must be an object.")
    settings = {
        section: {k: v for k, v in values.items() if k not in MACHINE_SETTINGS.get(section, ())}
        for section, values in settings.items()
    }
    artwork = data.get("artwork") or {}
    size = None
    if isinstance(artwork, dict) and isinstance(artwork.get("width"), int) and isinstance(artwork.get("height"), int):
        if artwork["width"] > 0 and artwork["height"] > 0:
            size = (artwork["width"], artwork["height"])
    try:
        profile = InspectionProfile(
            name=name.strip(),
            description=str(data.get("description", "")),
            sensitivity=sensitivity,
            settings=settings,
            regions=regions_from_data(data.get("regions", []), source),
            text_zones=zones_from_data(data.get("text_zones", []), source),
            barcode_zones=barcode_zones_from_data(data.get("barcode_zones", []), source),
            artwork_name=str(artwork.get("file_name") or "") if isinstance(artwork, dict) else "",
            artwork_size=size,
            created=str(data.get("created", "")),
            modified=str(data.get("modified", "")),
            app_version=str(data.get("application_version", "")),
            extra={k: v for k, v in data.items() if k not in _KNOWN_KEYS},
        )
    except (RegionError, TextZoneError, BarcodeZoneError) as exc:
        raise ProfileError(str(exc)) from exc
    try:
        parse_config(_merged(default_config_data(), profile))
    except ConfigError as exc:
        raise ProfileError(f"{source} has invalid settings: {exc}") from exc
    return profile


def save_profile(path: Path, profile: InspectionProfile) -> None:
    """Write the profile atomically. Raises OSError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(profile_to_data(profile), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_profile(path: Path) -> InspectionProfile:
    """Read and validate a profile file. Raises ProfileError."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"The profile file does not exist: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Profile file {path.name} cannot be read: {exc}") from exc
    return profile_from_data(data, f"Profile file {path.name}")


# ---------------------------------------------------------------- the profile library


def profiles_dir() -> Path:
    return user_data_dir() / "profiles"


def file_stem(name: str) -> str:
    """A safe Windows file name stem for a profile name."""
    stem = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._ ")[:60] or "profile"
    return stem + "_" if stem.lower() in _RESERVED else stem


@dataclass(frozen=True)
class ProfileEntry:
    path: Path
    profile: InspectionProfile | None
    error: str = ""  # why the file could not be loaded

    @property
    def name(self) -> str:
        return self.profile.name if self.profile is not None else self.path.stem


def list_profiles(directory: Path) -> list[ProfileEntry]:
    """Every profile file in ``directory`` (unreadable ones with their error), sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    entries = []
    for path in directory.glob(f"*{PROFILE_SUFFIX}"):
        try:
            entries.append(ProfileEntry(path, load_profile(path)))
        except ProfileError as exc:
            entries.append(ProfileEntry(path, None, str(exc)))
    return sorted(entries, key=lambda e: (e.name.casefold(), e.path.name))


def find_profile(directory: Path, name: str) -> ProfileEntry | None:
    return next((e for e in list_profiles(directory) if e.profile is not None and e.profile.name == name), None)


def path_for(directory: Path, name: str) -> Path:
    """The file for profile ``name``: its existing file, or a new unused file name."""
    existing = find_profile(directory, name)
    if existing is not None:
        return existing.path
    stem = file_stem(name)
    path = Path(directory) / f"{stem}{PROFILE_SUFFIX}"
    counter = 1
    while path.exists():
        counter += 1
        path = Path(directory) / f"{stem}_{counter}{PROFILE_SUFFIX}"
    return path


def delete_profile(path: Path) -> None:
    Path(path).unlink(missing_ok=True)


def shape_warning(profile: InspectionProfile, width: int, height: int) -> str | None:
    """A warning if the artwork's shape differs from the one the profile's zones were drawn on."""
    if profile.artwork_size is None or not (profile.regions or profile.text_zones or profile.barcode_zones):
        return None
    pw, ph = profile.artwork_size
    ratio = (width / height) / (pw / ph)
    if abs(ratio - 1.0) <= ASPECT_TOLERANCE:
        return None
    return (
        f"Profile '{profile.name}' was made for an artwork of {pw} × {ph} px; this artwork is {width} × {height} px "
        "with a different shape, so its regions and zones may be in the wrong place. Check them before inspecting."
    )
