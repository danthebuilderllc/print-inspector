"""Synchronised zoom and pan between image views showing the same label.

The views show the label at different resolutions (the artwork in artwork
pixels, the registered print and the difference maps on the comparison grid,
the perspective-corrected print at its own size). Each view is registered with
its scale relative to the artwork, ``(sx, sy)`` = view pixels per artwork pixel,
so the views can show the same part of the label at the same size on screen.
A view whose content has no fixed relation to the artwork (the original photo)
is given no scale and is left alone.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF

from print_inspector.gui.image_panel import ImageView


class ViewSynchronizer(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._scales: dict[ImageView, tuple[float, float] | None] = {}
        self._order: list[ImageView] = []
        self._enabled = True
        self._busy = False
        self._leader: ImageView | None = None

    def add_view(self, view: ImageView) -> None:
        self._scales[view] = None
        self._order.append(view)
        view.view_changed.connect(lambda v=view: self._on_view_changed(v))

    def set_scale(self, view: ImageView, scale: tuple[float, float] | None) -> None:
        """View pixels per artwork pixel, or None to exclude the view."""
        self._scales[view] = scale

    def scale(self, view: ImageView) -> tuple[float, float] | None:
        return self._scales.get(view)

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if enabled and self._leader is not None:
            self.follow(self._leader)

    def synced_views(self) -> list[ImageView]:
        """Views that currently take part (have an image and a scale)."""
        return [v for v in self._order if v.has_image() and self._scales[v] is not None]

    def leader(self) -> ImageView | None:
        """The view the user navigated last, if it still takes part; else the first one that does."""
        synced = self.synced_views()
        if self._leader in synced:
            return self._leader
        return synced[0] if synced else None

    def follow(self, source: ImageView) -> None:
        """Make the other synced views show what ``source`` shows."""
        scale = self._scales.get(source)
        if not self._enabled or self._busy or scale is None or not source.has_image():
            return
        centre, zoom = source.view_state()
        sx, sy = scale
        art_x, art_y = centre.x() / sx, centre.y() / sy
        art_zoom = zoom * (sx + sy) / 2  # screen pixels per artwork pixel
        self._busy = True
        try:
            for view in self.synced_views():
                if view is source:
                    continue
                vx, vy = self._scales[view]
                view.apply_view_state(QPointF(art_x * vx, art_y * vy), art_zoom / ((vx + vy) / 2))
        finally:
            self._busy = False

    def _on_view_changed(self, view: ImageView) -> None:
        if self._busy:
            return
        self._leader = view
        self.follow(view)
