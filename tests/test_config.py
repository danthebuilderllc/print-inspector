import json

import pytest

from print_inspector.config import (
    ConfigError,
    deep_merge,
    load_config,
    parse_config,
    resource_path,
    DEFAULT_CONFIG_RELATIVE_PATH,
)


def _defaults() -> dict:
    return json.loads(resource_path(DEFAULT_CONFIG_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_default_config_loads():
    config = load_config(user_path=None)
    assert config.app.name == "Print Inspector"
    assert ".png" in config.image_loading.original_extensions
    assert ".tiff" in config.image_loading.original_extensions
    assert config.image_loading.printed_extensions == (".jpg", ".jpeg", ".png")
    assert config.inspection.sensitivity == "medium"
    assert config.ui.zoom_step > 1.0
    assert config.ui.min_zoom < config.ui.max_zoom


def test_user_override_is_merged(tmp_path):
    user = tmp_path / "config.json"
    user.write_text(json.dumps({"inspection": {"sensitivity": "high"}}), encoding="utf-8")
    config = load_config(user_path=user)
    assert config.inspection.sensitivity == "high"
    assert config.ui.window_width == _defaults()["ui"]["window_width"]


def test_missing_user_override_is_ignored(tmp_path):
    config = load_config(user_path=tmp_path / "does_not_exist.json")
    assert config.inspection.sensitivity == "medium"


def test_invalid_json_reports_location(tmp_path):
    user = tmp_path / "config.json"
    user.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="line 1"):
        load_config(user_path=user)


def test_missing_default_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(default_path=tmp_path / "missing.json")


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("inspection", "sensitivity", "extreme", "sensitivity"),
        ("ui", "zoom_step", 1.0, "zoom_step"),
        ("ui", "max_zoom", 0.01, "max_zoom"),
        ("ui", "window_width", "wide", "type int"),
        ("ui", "window_width", True, "type int"),
        ("image_loading", "max_pixels", -5, "greater than 0"),
        ("image_loading", "printed_extensions", [], "at least one"),
        ("image_loading", "printed_extensions", ["jpg"], "Invalid extension"),
        ("logging", "level", "LOUD", "logging.level"),
    ],
)
def test_invalid_values_rejected(section, key, value, message):
    data = _defaults()
    data[section][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_unknown_key_rejected():
    data = _defaults()
    data["ui"]["zoom_stepp"] = 2.0
    with pytest.raises(ConfigError, match="zoom_stepp"):
        parse_config(data)


def test_missing_section_rejected():
    data = _defaults()
    del data["ui"]
    with pytest.raises(ConfigError, match="'ui'"):
        parse_config(data)


def test_extensions_are_normalised_to_lowercase():
    data = _defaults()
    data["image_loading"]["printed_extensions"] = [".JPG", ".jpg", ".Png"]
    config = parse_config(data)
    assert config.image_loading.printed_extensions == (".jpg", ".png")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}}
    assert base == {"a": {"x": 1, "y": 2}}


# ---------------------------------------------------------------- capture_quality section


def test_capture_quality_defaults_load():
    cq = load_config(user_path=None).capture_quality
    assert cq.min_megapixels_warning == 12.0
    assert cq.blur_warning_sigma_px < cq.blur_critical_sigma_px
    assert cq.min_pixels_per_mm_critical < cq.min_pixels_per_mm_warning


def test_capture_quality_user_override(tmp_path):
    user = tmp_path / "config.json"
    user.write_text(json.dumps({"capture_quality": {"blur_warning_sigma_px": 1.5}}), encoding="utf-8")
    assert load_config(user_path=user).capture_quality.blur_warning_sigma_px == 1.5


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("blur_warning_sigma_px", 5.0, "blur_warning_sigma_px' must be less than"),
        ("underexposure_critical_highlight", 200, "underexposure_critical_highlight' must be less than"),
        ("min_pixels_per_mm_critical", 20, "min_pixels_per_mm_critical' must be less than"),
        ("overexposure_critical_fraction", 1.5, "between 0 and 1"),
        ("clip_level", 300, "between 0 and 255"),
        ("blur_contrast_window_px", 30, "odd number"),
        ("blur_sample_tiles", 0, "greater than 0"),
        ("blur_sample_tiles", 2.5, "type int"),
        ("border_band_fraction", 0.3, "below 0.25"),
    ],
)
def test_invalid_capture_quality_values_rejected(key, value, message):
    data = _defaults()
    data["capture_quality"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_missing_capture_quality_key_rejected():
    data = _defaults()
    del data["capture_quality"]["clip_level"]
    with pytest.raises(ConfigError, match="capture_quality.clip_level"):
        parse_config(data)


def test_unknown_extension_rejected():
    data = _defaults()
    data["image_loading"]["original_extensions"] = [".png", ".gif"]
    with pytest.raises(ConfigError, match="Unsupported extension"):
        parse_config(data)


# ---------------------------------------------------------------- alignment section


def test_alignment_defaults_load():
    al = load_config(user_path=None).alignment
    assert al.interpolation == "cubic"
    assert al.min_edge_support_critical < al.min_edge_support_warning
    assert al.residual_warning_fraction < al.residual_critical_fraction


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("interpolation", "nearest", "interpolation"),
        ("interpolation", 3, "type str"),
        ("min_edge_support_warning", 0.5, "min_edge_support_critical' must be less than"),
        ("residual_critical_fraction", 0.001, "residual_warning_fraction' must be less than"),
        ("min_similarity_warning", 1.5, "between 0 and 1"),
        ("max_corner_angle_deviation_deg_critical", 95, "below 90"),
        ("max_side_ratio_warning", 1.0, "greater than 1"),
        ("residual_min_tiles", 3, "at least 4"),
        ("feature_min_inliers", 0, "greater than 0"),
    ],
)
def test_invalid_alignment_values_rejected(key, value, message):
    data = _defaults()
    data["alignment"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_alignment_interpolation_case_insensitive():
    data = _defaults()
    data["alignment"]["interpolation"] = "LANCZOS"
    assert parse_config(data).alignment.interpolation == "lanczos"


# ---------------------------------------------------------------- registration section


def test_registration_defaults_load():
    reg = load_config(user_path=None).registration
    assert reg.residual_p95_warning_px < reg.residual_p95_critical_px
    assert reg.coverage_critical < reg.coverage_warning
    assert reg.ecc_gauss_filter_size % 2 == 1


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("interpolation", "area", "interpolation"),
        ("ecc_gauss_filter_size", 4, "odd number"),
        ("tile_size_px", 16, "at least 32"),
        ("local_poly_degree", 7, "5 or less"),
        ("residual_p95_critical_px", 0.3, "residual_p95_warning_px' must be less than"),
        ("coverage_warning", 0.1, "coverage_critical' must be less than"),
        ("feature_min_spread", 1.5, "between 0 and 1"),
        ("tile_max_shift_px", 0, "greater than 0"),
    ],
)
def test_invalid_registration_values_rejected(key, value, message):
    data = _defaults()
    data["registration"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


# ---------------------------------------------------------------- comparison section


def test_comparison_defaults_load():
    cmp_ = load_config(user_path=None).comparison
    assert cmp_.fine_low_sigma < cmp_.fine_high_sigma
    assert cmp_.sensitivity_high_factor < 1.0 < cmp_.sensitivity_low_factor


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("fine_low_sigma", 9.0, "fine_low_sigma' must be less than"),
        ("coarse_levels", 5, "between 0 and 4"),
        ("tile_size_px", 64, "at least 128"),
        ("tile_margin_px", 400, "quarter of the tile size"),
        ("contrast_window_px", 8, "odd number"),
        ("contrast_confirm_ratio", 0.9, "greater than 1"),
        ("sensitivity_high_factor", 1.2, "sensitivity_high_factor"),
        ("max_edge_jitter_px", 0.1, "at least"),
        ("min_model_similarity_critical", 0.9, "min_model_similarity_critical' must be less than"),
    ],
)
def test_invalid_comparison_values_rejected(key, value, message):
    data = _defaults()
    data["comparison"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_comparison_zero_coarse_levels_allowed():
    data = _defaults()
    data["comparison"]["coarse_levels"] = 0
    assert parse_config(data).comparison.coarse_levels == 0


# ---------------------------------------------------------------- defects and decision sections


def test_defects_and_decision_defaults_load():
    config = load_config(user_path=None)
    assert config.defects.severity_major_size_mm < config.defects.severity_critical_size_mm
    assert config.decision.fail_min_severity == "major"
    assert config.decision.allow_pass_with_warnings is True


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("defects", "severity_critical_size_mm", 0.1, "severity_major_size_mm' must be less than"),
        ("defects", "likely_min_fraction", 1.5, "between 0 and 1"),
        ("defects", "merge_distance_mm", 0, "greater than 0"),
        ("decision", "fail_min_severity", "fatal", "fail_min_severity"),
        ("decision", "max_minor_defects", -1, "0 or greater"),
        ("decision", "allow_pass_with_warnings", "yes", "type bool"),
        ("decision", "critical_region_severity", "severe", "critical_region_severity"),
    ],
)
def test_invalid_defect_and_decision_values_rejected(section, key, value, message):
    data = _defaults()
    data[section][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_text_defaults_load():
    text = load_config(user_path=None).text
    assert text.enabled and text.required and text.languages == "eng"
    assert text.critical_min_confidence >= text.min_word_confidence
    assert text.critical_partial_min_fraction <= text.partial_min_fraction
    assert text.severity_incorrect_number == "critical"


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("languages", "eng;rm -rf", "language codes"),
        ("languages", "", "language codes"),
        ("line_psm", 14, "page segmentation mode"),
        ("min_word_confidence", 120.0, "between 0 and 100"),
        ("critical_min_confidence", 10.0, "at least"),
        ("partial_min_fraction", 0.0, "between 0 and 1"),
        ("critical_partial_min_fraction", 0.5, "must not exceed"),
        ("severity_missing_word", "fatal", "one of"),
        ("target_line_height_px", 8, "at least 16"),
        ("timeout_s", 0.0, "positive|greater"),
        ("enabled", "yes", "bool"),
    ],
)
def test_invalid_text_values_rejected(key, value, message):
    data = _defaults()
    data["text"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(data)


def test_text_severity_case_is_normalised():
    data = _defaults()
    data["text"]["severity_extra_text"] = "MAJOR"
    assert parse_config(data).text.severity_extra_text == "major"


def test_report_settings_load_and_validate():
    config = load_config(user_path=None)
    assert config.report.pdf is True and config.report.title == "Print Inspection Report"
    assert config.report.inspector == "" and config.report.max_evidence_crops > 0
    assert config.decision.critical_region_severity == "major"
    data = _defaults()
    data["report"]["title"] = "  "
    assert parse_config(data).report.title == "Print Inspection Report"
    data["report"]["max_evidence_crops"] = -1
    with pytest.raises(ConfigError, match="0 or greater"):
        parse_config(data)
    data = _defaults()
    data["report"]["pdf"] = "yes"
    with pytest.raises(ConfigError, match="bool"):
        parse_config(data)


def test_config_overrides_are_the_differences_from_the_defaults():
    import dataclasses

    from print_inspector.config import config_overrides, config_with_overrides

    config = load_config(user_path=None)
    assert config_overrides(config) == {}
    changed = dataclasses.replace(config, report=dataclasses.replace(config.report, inspector="A. B."))
    overrides = config_overrides(changed)
    assert overrides == {"report": {"inspector": "A. B."}}
    assert config_with_overrides(overrides) == changed
