"""Critical text zone files."""
import json

import pytest

from print_inspector.core.text_zones import TextZone, TextZoneError, load_zones, save_zones, zones_path


def test_path_next_to_artwork(tmp_path):
    assert zones_path(tmp_path / "label.png") == tmp_path / "label.png.zones.json"


def test_round_trip_and_removal(tmp_path):
    path = zones_path(tmp_path / "label.png")
    assert load_zones(path) == ()
    zones = (
        TextZone("Lot", 0.1, 0.2, 0.3, 0.1, "LOT 12345"),
        TextZone("Ingredients", 0.0, 0.5, 1.0, 0.5, critical=False),
    )
    save_zones(path, zones)
    assert load_zones(path) == zones
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["zones"][0]["expected_text"] == "LOT 12345"
    save_zones(path, ())
    assert not path.exists()


def test_rect_and_contains():
    zone = TextZone("z", 0.25, 0.5, 0.5, 0.25)
    assert zone.rect(400, 200) == (100, 100, 200, 50)
    assert zone.contains(150, 120, 400, 200) and not zone.contains(50, 120, 400, 200)


@pytest.mark.parametrize("raw, message", [
    ({"zones": [{"name": "", "x": 0, "y": 0, "width": 0.1, "height": 0.1}]}, "no name"),
    ({"zones": [{"name": "a", "x": 0.95, "y": 0, "width": 0.1, "height": 0.1}]}, "beyond"),
    ({"zones": [{"name": "a", "x": 0, "y": 0, "width": 0, "height": 0.1}]}, "outside"),
    ({"zones": [{"name": "a", "x": 0}]}, "incomplete"),
    ({"zone": []}, "no 'zones' list"),
])
def test_invalid_files(tmp_path, raw, message):
    path = tmp_path / "a.png.zones.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TextZoneError, match=message):
        load_zones(path)


def test_unreadable_json(tmp_path):
    path = tmp_path / "a.png.zones.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TextZoneError, match="cannot be read"):
        load_zones(path)
