"""Settings dialog and the user settings file."""
import json

import pytest

from print_inspector.config import (
    ConfigError,
    config_with_overrides,
    load_config,
    read_user_overrides,
    write_user_overrides,
)
from print_inspector.gui.settings_dialog import SETTING_FIELDS, SettingsDialog, config_value, merge_values


def test_every_field_exists_in_the_config(app_config):
    for f in SETTING_FIELDS:
        config_value(app_config, f)  # raises AttributeError for a wrong section/key


def test_overrides_round_trip(tmp_path):
    path = tmp_path / "sub" / "config.json"
    assert read_user_overrides(path) == {}
    write_user_overrides(path, {"decision": {"max_minor_defects": 7}})
    assert read_user_overrides(path) == {"decision": {"max_minor_defects": 7}}
    assert not path.with_name("config.json.tmp").exists()
    assert config_with_overrides(read_user_overrides(path)).decision.max_minor_defects == 7
    assert load_config(user_path=path).decision.max_minor_defects == 7


def test_invalid_overrides_rejected():
    with pytest.raises(ConfigError):
        config_with_overrides({"defects": {"severity_major_size_mm": 5.0, "severity_critical_size_mm": 1.0}})


def test_merge_keeps_other_settings_and_drops_defaults(app_config):
    existing = {"logging": {"level": "DEBUG"}, "decision": {"max_minor_defects": 9}}
    values = {(f.section, f.key): config_value(app_config, f) for f in SETTING_FIELDS}
    values[("decision", "fail_min_severity")] = "critical"
    merged = merge_values(existing, values, app_config)
    # The unlisted logging override stays; max_minor back at its default is dropped.
    assert merged == {"logging": {"level": "DEBUG"}, "decision": {"fail_min_severity": "critical"}}
    assert existing["decision"] == {"max_minor_defects": 9}  # input not modified


def test_dialog_shows_current_values_and_returns_changes(qtbot, app_config, tmp_path):
    dialog = SettingsDialog(app_config, {}, tmp_path / "config.json")
    qtbot.addWidget(dialog)
    editors = dialog.editors
    assert editors[("decision", "fail_min_severity")].currentData() == app_config.decision.fail_min_severity
    assert editors[("defects", "severity_major_size_mm")].value() == pytest.approx(
        app_config.defects.severity_major_size_mm
    )
    editors[("decision", "max_minor_defects")].setValue(0)
    editors[("defects", "severity_major_size_mm")].setValue(0.3)
    editors[("decision", "allow_pass_with_warnings")].setChecked(False)
    dialog.accept()
    assert dialog.result() == SettingsDialog.DialogCode.Accepted
    assert dialog.overrides() == {
        "decision": {"max_minor_defects": 0, "allow_pass_with_warnings": False},
        "defects": {"severity_major_size_mm": 0.3},
    }
    config = dialog.result_config()
    assert config.decision.max_minor_defects == 0 and config.defects.severity_major_size_mm == 0.3


def test_dialog_rejects_inconsistent_values(qtbot, app_config):
    dialog = SettingsDialog(app_config, {}, None)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.editors[("defects", "severity_major_size_mm")].setValue(5.0)
    dialog.editors[("defects", "severity_critical_size_mm")].setValue(1.0)
    dialog.accept()
    assert dialog.result_config() is None
    assert dialog.isVisible()
    assert "severity_major_size_mm" in dialog.error_label.text()


def test_restore_defaults(qtbot, app_config):
    changed = config_with_overrides({"decision": {"max_minor_defects": 11}})
    dialog = SettingsDialog(changed, {"decision": {"max_minor_defects": 11}}, None)
    qtbot.addWidget(dialog)
    assert dialog.editors[("decision", "max_minor_defects")].value() == 11
    dialog.defaults_button.click()
    assert dialog.editors[("decision", "max_minor_defects")].value() == app_config.decision.max_minor_defects
    dialog.accept()
    assert dialog.overrides() == {}


def test_written_file_is_plain_json(tmp_path):
    path = tmp_path / "config.json"
    write_user_overrides(path, {"capture_quality": {"assumed_label_long_side_mm": 80.0}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"capture_quality": {"assumed_label_long_side_mm": 80.0}}
