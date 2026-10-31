"""The application icon, drawn in code (no image files to bundle or lose).

``packaging/make_icon.py`` renders the same drawing into the .ico file of the
Windows executable.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def icon_image(size: int) -> QImage:
    """A label with print lines under a magnifying glass with a check mark, ``size`` x ``size`` pixels."""
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    s = size / 64.0
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#1e3a5f"))
    p.drawRoundedRect(QRectF(1 * s, 1 * s, 62 * s, 62 * s), 12 * s, 12 * s)
    p.setBrush(QColor("#f8fafc"))
    p.drawRoundedRect(QRectF(9 * s, 11 * s, 34 * s, 42 * s), 3 * s, 3 * s)
    p.setBrush(QColor("#94a3b8"))
    for i, width in enumerate((24, 18, 22, 14)):
        p.drawRect(QRectF(14 * s, (18 + 7 * i) * s, width * s, 3 * s))
    centre, radius = QPointF(40 * s, 38 * s), 13 * s
    p.setBrush(QColor(255, 255, 255, 215))
    p.setPen(QPen(QColor("#f59e0b"), max(1.0, 4 * s)))
    p.drawEllipse(centre, radius, radius)
    pen = QPen(QColor("#f59e0b"), max(1.5, 6 * s), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(49.5 * s, 47.5 * s), QPointF(57 * s, 55 * s))
    tick = QPainterPath(QPointF(34 * s, 38.5 * s))
    tick.lineTo(38.5 * s, 43 * s)
    tick.lineTo(46.5 * s, 33.5 * s)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(QColor("#16a34a"), max(1.0, 3.5 * s), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                  Qt.PenJoinStyle.RoundJoin))
    p.drawPath(tick)
    p.end()
    return image


def app_icon() -> QIcon:
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(QPixmap.fromImage(icon_image(size)))
    return icon
