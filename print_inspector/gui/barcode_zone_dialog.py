"""Draw and edit the barcode inspection zones of an artwork."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from print_inspector.config import UIConfig
from print_inspector.core.barcode_zones import ANY_FORMAT, BarcodeZone, BarcodeZoneError, validate_zones
from print_inspector.core.barcodes import FORMAT_LABELS
from print_inspector.gui.zone_dialog import Column, ZoneEditorDialog

COL_NAME, COL_FORMAT, COL_EXPECTED, COL_CRITICAL = 0, 1, 2, 3


class BarcodeZoneDialog(ZoneEditorDialog):
    """Mark where barcodes must be; optionally their type and exact value."""

    columns = (
        Column("Name", "text"),
        Column("Type", "choice", (("Any", ANY_FORMAT),) + tuple((label, key) for key, label in FORMAT_LABELS.items()),
               default=ANY_FORMAT),
        Column("Expected value (optional)", "text", stretch=True),
        Column("Critical", "check", default=True),
    )
    hint_text = (
        "A barcode must be found and decoded in each zone. Give the expected value (e.g. the GTIN from the "
        "product data) to check the content against it instead of against the artwork; EAN-13 / UPC-A values "
        "need a correct check digit. Findings in critical zones are critical, and uncertain ones give REVIEW. "
        "Zones are saved next to the artwork file and used for every inspection of it."
    )

    def __init__(
        self, artwork: np.ndarray, zones: tuple[BarcodeZone, ...], ui_config: UIConfig, parent: QWidget | None = None
    ) -> None:
        super().__init__("Barcode Zones", artwork, ui_config, parent)
        for zone in zones:
            self.add_row(self.rect_of(zone.x, zone.y, zone.width, zone.height),
                         [zone.name, zone.format, zone.expected_value, zone.critical])
        self.draw_button.setChecked(not zones)

    def zones(self) -> tuple[BarcodeZone, ...]:
        return tuple(
            BarcodeZone(name, *rect, fmt, expected, critical)
            for rect, (name, fmt, expected, critical) in self.rows()
        )

    def validate(self) -> str | None:
        message = super().validate()
        if message:
            return message
        try:
            validate_zones(self.zones())
        except BarcodeZoneError as exc:
            return str(exc)
        return None
