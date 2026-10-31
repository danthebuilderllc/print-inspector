"""Inspection profiles: what they store, validation, applying them, and the local profile library."""
import dataclasses
import json

import pytest

from print_inspector.config import config_data, load_config
from print_inspector.core.barcode_zones import BarcodeZone
from print_inspector.core.profiles import (
    FILE_FORMAT,
    PROFILE_SECTIONS,
    ProfileError,
    apply_profile,
    create_profile,
    delete_profile,
    file_stem,
    find_profile,
    is_profile_setting,
    list_profiles,
    load_profile,
    matches,
    path_for,
    profile_from_data,
    profile_to_data,
    profiles_dir,
    save_profile,
    shape_warning,
)
from print_inspector.core.regions import InspectionRegion, RegionKind
from print_inspector.core.text_zones import TextZone

REGIONS = (
    InspectionRegion("Logo", RegionKind.CRITICAL, 0.05, 0.05, 0.3, 0.2),
    InspectionRegion("Variable data", RegionKind.IGNORE, 0.6, 0.7, 0.35, 0.25),
)
TEXT_ZONES = (TextZone("Lot", 0.1, 0.8, 0.3, 0.1, expected_text="LOT 12345"),)
BARCODE_ZONES = (BarcodeZone("GTIN", 0.5, 0.1, 0.4, 0.3, "ean13", "5901234123457"),)


@pytest.fixture
def base():
    return load_config(user_path=None)


@pytest.fixture
def custom(base):
    """Settings for a demanding label: stricter acceptance, no barcodes, tighter colour tolerance."""
    return dataclasses.replace(
        base,
        decision=dataclasses.replace(base.decision, max_minor_defects=0, critical_region_severity="critical"),
        defects=dataclasses.replace(base.defects, severity_major_size_mm=0.3),
        text=dataclasses.replace(base.text, numbers_are_critical=False, min_word_confidence=70.0,
                                 tesseract_path="C:/Tools/Tesseract"),
        barcodes=dataclasses.replace(base.barcodes, enabled=False),
        color=dataclasses.replace(base.color, patch_tolerance_delta_e=3.0, calibration_profile="C:/x/profile.json"),
    )


def make(custom, name="Product 4711 front", **kwargs):
    values = dict(regions=REGIONS, text_zones=TEXT_ZONES, barcode_zones=BARCODE_ZONES,
                  artwork_name="front_v3.png", artwork_size=(2400, 1600))
    values.update(kwargs)
    return create_profile(name, "Front label, 100 x 70 mm", custom, "high", **values)


def test_profile_stores_the_label_settings_but_not_machine_paths(custom):
    profile = make(custom)
    assert set(profile.settings) == set(PROFILE_SECTIONS)
    assert profile.settings["decision"]["max_minor_defects"] == 0
    assert profile.settings["defects"]["severity_major_size_mm"] == 0.3
    assert profile.settings["text"]["min_word_confidence"] == 70.0
    assert profile.settings["barcodes"]["enabled"] is False
    assert profile.settings["color"]["patch_tolerance_delta_e"] == 3.0
    for section, key in (("text", "tesseract_path"), ("text", "tessdata_dir"), ("color", "calibration_profile")):
        assert key not in profile.settings[section] and not is_profile_setting(section, key)
    assert is_profile_setting("decision", "fail_min_severity") and is_profile_setting("inspection", "sensitivity")
    assert not is_profile_setting("capture_quality", "blur_warning_sigma_px")
    assert profile.created == profile.modified and profile.app_version == custom.app.version
    assert "1 critical / 1 ignored region(s)" in profile.summary() and "sensitivity high" in profile.summary()


def test_apply_profile_keeps_the_computers_paths_and_other_sections(base, custom):
    profile = make(custom)
    machine = dataclasses.replace(
        base,
        text=dataclasses.replace(base.text, tesseract_path="D:/OCR"),
        capture_quality=dataclasses.replace(base.capture_quality, min_megapixels_warning=8.0),
    )
    applied = apply_profile(profile, machine)
    assert applied.inspection.sensitivity == "high"
    assert applied.decision == custom.decision and applied.defects == custom.defects
    assert applied.barcodes == custom.barcodes
    assert applied.text.min_word_confidence == 70.0 and applied.text.tesseract_path == "D:/OCR"
    assert applied.color.patch_tolerance_delta_e == 3.0 and applied.color.calibration_profile == ""
    assert applied.capture_quality.min_megapixels_warning == 8.0  # not part of a profile
    assert matches(profile, applied, "high", REGIONS, TEXT_ZONES, BARCODE_ZONES)
    assert not matches(profile, applied, "medium", REGIONS, TEXT_ZONES, BARCODE_ZONES)
    assert not matches(profile, applied, "high", REGIONS[:1], TEXT_ZONES, BARCODE_ZONES)
    changed = dataclasses.replace(applied, color=dataclasses.replace(applied.color, patch_tolerance_delta_e=4.0))
    assert not matches(profile, changed, "high", REGIONS, TEXT_ZONES, BARCODE_ZONES)


def test_save_and_load_round_trip(tmp_path, custom):
    profile = make(custom)
    path = tmp_path / "p.json"
    save_profile(path, profile)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == FILE_FORMAT and data["artwork"] == {"file_name": "front_v3.png", "width": 2400,
                                                                 "height": 1600}
    assert data["barcode_zones"][0]["expected_value"] == "5901234123457"
    assert data["text_zones"][0]["expected_text"] == "LOT 12345"
    assert data["regions"][1]["kind"] == "ignore"
    loaded = load_profile(path)
    assert loaded == profile
    assert loaded.regions == REGIONS and loaded.text_zones == TEXT_ZONES and loaded.barcode_zones == BARCODE_ZONES


def test_older_profile_without_new_settings_still_loads(custom, base):
    data = profile_to_data(make(custom))
    del data["settings"]["color"]["severity_global_shift"]  # a setting added after the profile was saved
    data["future_note"] = "kept"
    profile = profile_from_data(data)
    applied = apply_profile(profile, base)
    assert applied.color.severity_global_shift == base.color.severity_global_shift
    assert profile_to_data(profile)["future_note"] == "kept"


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(format="something else"), "not a Print Inspector profile"),
    (lambda d: d.update(version=99), "newer version"),
    (lambda d: d.update(name=""), "no name"),
    (lambda d: d.update(sensitivity="extreme"), "sensitivity"),
    (lambda d: d["settings"].update(capture_quality={}), "cannot change: capture_quality"),
    (lambda d: d["settings"]["decision"].update(fail_min_severity="huge"), "invalid settings"),
    (lambda d: d["settings"]["defects"].update(no_such_key=1), "invalid settings"),
    (lambda d: d["regions"][0].update(kind="sometimes"), "unknown type"),
    (lambda d: d["barcode_zones"][0].update(expected_value="5901234123458"), "check digit"),
    (lambda d: d["text_zones"][0].update(x=1.5), "outside"),
])
def test_invalid_profiles_are_rejected(custom, change, message):
    data = profile_to_data(make(custom))
    change(data)
    with pytest.raises(ProfileError, match=message):
        profile_from_data(data)


def test_unreadable_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ProfileError, match="cannot be read"):
        load_profile(path)
    with pytest.raises(ProfileError, match="does not exist"):
        load_profile(tmp_path / "missing.json")


def test_machine_paths_in_a_profile_file_are_ignored(custom, base):
    data = profile_to_data(make(custom))
    data["settings"]["text"]["tesseract_path"] = "E:/evil"
    profile = profile_from_data(data)
    assert "tesseract_path" not in profile.settings["text"]
    assert apply_profile(profile, base).text.tesseract_path == base.text.tesseract_path


def test_library(tmp_path, custom):
    directory = tmp_path / "profiles"
    assert list_profiles(directory) == []
    first = make(custom, "Front: 4711/A")
    path = path_for(directory, first.name)
    assert path.name == "Front_4711_A.json"
    save_profile(path, first)
    other = make(custom, "Front 4711 A")  # a different name with the same file stem
    other_path = path_for(directory, other.name)
    assert other_path.name == "Front_4711_A_2.json"
    save_profile(other_path, other)
    (directory / "broken.json").write_text("[]", encoding="utf-8")
    entries = list_profiles(directory)
    assert [e.name for e in entries] == ["broken", "Front 4711 A", "Front: 4711/A"]
    assert entries[0].profile is None and "not a Print Inspector profile" in entries[0].error
    assert find_profile(directory, "Front: 4711/A").path == path
    assert path_for(directory, "Front: 4711/A") == path  # saving again replaces the same file
    delete_profile(path)
    assert find_profile(directory, "Front: 4711/A") is None
    assert file_stem("con") == "con_" and file_stem("  ") == "profile"
    assert profiles_dir().name == "profiles"


def test_shape_warning(custom):
    profile = make(custom)
    assert shape_warning(profile, 1200, 800) is None  # same shape, other resolution
    assert "different shape" in shape_warning(profile, 1200, 1200)
    assert shape_warning(make(custom, regions=(), text_zones=(), barcode_zones=()), 1200, 1200) is None


def test_config_data_round_trip(base):
    from print_inspector.config import parse_config

    assert parse_config(config_data(base)) == base


def test_profile_needs_a_name(custom):
    with pytest.raises(ProfileError, match="name"):
        make(custom, "   ")


def test_resaving_keeps_the_creation_date(custom):
    first = dataclasses.replace(make(custom), created="2026-01-01T08:00:00+01:00")
    second = create_profile(first.name, first.description, custom, "low", previous=first)
    assert second.created == first.created and second.sensitivity == "low"
