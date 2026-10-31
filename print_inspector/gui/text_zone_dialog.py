"""Draw and edit the critical text zones of an artwork."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from print_inspector.config import UIConfig
from print_inspector.core.text_zones import TextZone
from print_inspector.gui.zone_dialog import Column, ZoneEditorDialog

COL_NAME, COL_EXPECTED, COL_CRITICAL = 0, 1, 2


class TextZoneDialog(ZoneEditorDialog):
    """Mark zones whose text must be checked strictly; optionally give their exact text."""

    columns = (
        Column("Name", "text"),
        Column("Expected text (optional)", "text", stretch=True),
        Column("Critical", "check", default=True),
    )
    hint_text = (
        "Text in a critical zone is checked strictly: higher OCR confidence is required, smaller ink "
        "losses count, and every finding is critical. If OCR cannot read it, the label goes to REVIEW. "
        "Give the exact text to compare the print with it instead of with the artwork. "
        "Zones are saved next to the artwork file and used for every inspection of it."
    )

    def __init__(
        self, artwork: np.ndarray, zones: tuple[TextZone, ...], ui_config: UIConfig, parent: QWidget | None = None
    ) -> None:
        super().__init__("Critical Text Zones", artwork, ui_config, parent)
        for zone in zones:
            self.add_row(self.rect_of(zone.x, zone.y, zone.width, zone.height),
                         [zone.name, zone.expected_text, zone.critical])
        self.draw_button.setChecked(not zones)

    def zones(self) -> tuple[TextZone, ...]:
        return tuple(TextZone(name, *rect, expected, critical) for rect, (name, expected, critical) in self.rows())
