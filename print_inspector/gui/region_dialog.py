"""Draw and edit the critical and ignored regions of an artwork."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget

from print_inspector.config import UIConfig
from print_inspector.core.regions import InspectionRegion, RegionError, RegionKind, validate_regions
from print_inspector.gui.zone_dialog import Column, ZoneEditorDialog

COL_NAME, COL_KIND = 0, 1


class RegionDialog(ZoneEditorDialog):
    """Mark areas to inspect strictly (critical) or not at all (ignored)."""

    columns = (
        Column("Name", "text", stretch=True),
        Column("Type", "choice", tuple((kind.label, kind.value) for kind in RegionKind),
               default=RegionKind.CRITICAL.value),
    )
    item_name = "Region"
    hint_text = (
        "Critical: a defect or colour finding touching the region is at least of the severity set in Settings "
        "(major by default, so it fails the label) and an uncertain one gives REVIEW. "
        "Ignored: the region is not inspected at all (no defects, text, colour or barcodes found there) - use it "
        "only for areas that may legitimately differ, such as variable data printed later. "
        "For text and barcodes use their own zones. Regions are saved next to the artwork, or in the profile "
        "if one is in use."
    )

    def __init__(
        self, artwork: np.ndarray, regions: tuple[InspectionRegion, ...], ui_config: UIConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Inspection Regions", artwork, ui_config, parent)
        self.draw_button.setText("Draw Region")
        self.delete_button.setText("Delete Region")
        for region in regions:
            self.add_row(self.rect_of(region.x, region.y, region.width, region.height),
                         [region.name, region.kind.value])
        self.draw_button.setChecked(not regions)

    def regions(self) -> tuple[InspectionRegion, ...]:
        return tuple(
            InspectionRegion(name, RegionKind(kind), *rect)
            for rect, (name, kind) in self.rows()
        )

    def validate(self) -> str | None:
        message = super().validate()
        if message:
            return message
        try:
            validate_regions(self.regions())
        except RegionError as exc:
            return str(exc)
        return None
