"""Check-stage blocks for the side panel, and text formatting of check metrics."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from print_inspector.core.alignment import AlignmentMetrics
from print_inspector.core.capture_quality import QualityMetrics
from print_inspector.core.comparison import ComparisonMetrics
from print_inspector.core.image_io import ImageRole, LoadedImage
from print_inspector.core.models import QualityIssue, QualityLevel, Severity
from print_inspector.core.registration import RegistrationMetrics
from print_inspector.gui.status_area import LEVEL_STYLE

SEVERITY_COLOURS = {Severity.CRITICAL: "#b91c1c", Severity.WARNING: "#92400e"}


def describe_image(image: LoadedImage) -> str:
    """Multi-line description: name, dimensions, file type and relevant metadata."""
    lines = [image.path.name, image.summary()]
    if image.exif_orientation != 1:
        lines.append(f"Rotated upright using EXIF orientation {image.exif_orientation}.")
    if image.has_alpha:
        lines.append("Transparency shown over white label stock.")
    if image.role is ImageRole.ORIGINAL and image.dpi is not None:
        size = image.physical_size_mm
        assert size is not None
        lines.append(f"{image.dpi[0]:.0f} dpi → {size[0]:.1f} × {size[1]:.1f} mm")
    lines.extend(image.notes)
    return "\n".join(lines)


def opt(value, fmt: str) -> str:
    return "n/a" if value is None else format(value, fmt)


def format_quality_metrics(m: QualityMetrics) -> str:
    lines = [f"Image        {m.image_width} × {m.image_height} px ({m.megapixels:.1f} MP)"]
    if m.content_region is not None:
        r = m.content_region
        lines.append(f"Label area   {r.width} × {r.height} px at ({r.x}, {r.y})")
    lines.append(f"Edge blur    {opt(m.blur_sigma_px, '.2f')} px")
    if m.pixels_per_mm is not None:
        basis = "assumed size" if m.label_size_assumed else "artwork size"
        lines.append(
            f"Resolution   {m.pixels_per_mm:.1f} px/mm, effective "
            f"{opt(m.effective_pixels_per_mm, '.1f')} ({basis} {m.label_long_side_mm:.0f} mm)"
        )
    lines.append(f"Blown out    {opt(m.clipped_fraction, '.1%')} (largest spot {opt(m.largest_glare_fraction, '.2%')})")
    lines.append(f"Highlights   {opt(m.highlight_level, '.0f')}/255, near-black {opt(m.dark_fraction, '.1%')}")
    lines.append(f"Light field  {opt(m.lighting_unevenness, '.0%')} variation")
    if m.border_content_ratios:
        lines.append(f"Border       content ratio {max(m.border_content_ratios.values()):.2f} (max side)")
    return "\n".join(lines)


def format_alignment_metrics(m: AlignmentMetrics) -> str:
    lines = [f"Method       {m.method.label if m.method else 'none'}"]
    if m.edge_support is not None:
        support = " ".join(f"{s:.0%}" for s in m.edge_support)
        lines.append(f"Edge found   {support} (top right bottom left)")
        lines.append(f"Edge fit     {opt(m.edge_fit_rms_px, '.2f')} px RMS")
    if m.feature_inliers is not None:
        lines.append(
            f"Artwork match {m.feature_inliers} points ({opt(m.feature_inlier_ratio, '.0%')} inliers, "
            f"{opt(m.feature_rms_px, '.1f')} px)"
        )
    if m.corner_angle_deviation_deg is not None:
        lines.append(f"Geometry     angles ±{m.corner_angle_deviation_deg:.0f}°, side ratio {opt(m.side_ratio, '.2f')}")
    if m.estimated_aspect is not None:
        target = f", artwork {m.target_aspect:.3f} ({opt(m.aspect_error, '.1%')} off)" if m.aspect_error is not None else ""
        lines.append(f"Proportions  {m.estimated_aspect:.3f}{target}")
    if m.similarity is not None:
        lines.append(f"Orientation  {m.quarter_turns * 90}° turn, margin {opt(m.orientation_margin, '.2f')}")
        lines.append(f"Similarity   {m.similarity:.2f}")
        lines.append(f"Residual     {opt(m.residual_fraction, '.2%')} of size ({m.residual_tiles} areas)")
    if m.artwork_scale is not None:
        lines.append(f"Artwork fit  scale x {m.artwork_scale[0]:.3f}, y {m.artwork_scale[1]:.3f}")
    if m.output_size is not None:
        scale = f", {m.scale_to_artwork:.2f}× artwork" if m.scale_to_artwork else ""
        lines.append(f"Output       {m.output_size[0]} × {m.output_size[1]} px{scale}")
    return "\n".join(lines)


def format_registration_metrics(m: RegistrationMetrics) -> str:
    lines = [f"Method       {m.method.label if m.method else 'none'}"]
    if m.grid_size is not None:
        lines.append(f"Grid         {m.grid_size[0]} × {m.grid_size[1]} px ({opt(m.grid_scale, '.2f')}× artwork)")
    if m.feature_matches is not None:
        used = "used" if m.features_used else "not used"
        lines.append(f"Features     {m.feature_inliers or 0}/{m.feature_matches} inliers, {used}")
    if m.ecc_model is not None:
        lines.append(f"ECC          {m.ecc_model}, correlation {opt(m.ecc_correlation, '.3f')}")
    lines.append(f"Photo blur   {opt(m.blur_match_sigma_px, '.2f')} px (matched)")
    if m.local_correction_applied:
        lines.append(
            f"Local fix    degree {m.local_degree}, up to {opt(m.local_max_correction_px, '.2f')} px "
            f"({opt(m.local_rms_before_px, '.2f')} → {opt(m.local_rms_after_px, '.2f')} px held-out)"
        )
    else:
        lines.append("Local fix    not needed")
    lines.append(
        f"Residual     median {opt(m.residual_median_px, '.2f')}, 95% {opt(m.residual_p95_px, '.2f')}, "
        f"max {opt(m.residual_max_px, '.2f')} px"
    )
    lines.append(f"Verified     {m.verified_tiles}/{m.structured_tiles} detailed areas ({opt(m.coverage, '.0%')})")
    lines.append(f"Similarity   {opt(m.structure_similarity, '.2f')}")
    lines.append(f"Compared     {opt(m.mask_fraction, '.0%')} of the artwork area")
    return "\n".join(lines)


def format_comparison_metrics(m: ComparisonMetrics) -> str:
    noise = "/".join(f"{v:.1f}" for v in m.noise_sigma)
    return "\n".join([
        f"Candidates   {m.candidate_count} ({m.candidate_area_fraction:.2%} of the label)"
        + (f", {m.candidates_dropped} weaker not listed" if m.candidates_dropped else ""),
        f"Sensitivity  {m.sensitivity} (thresholds x{m.threshold_factor:.2f})",
        f"Noise        {noise} gray levels (B/G/R), texture up to {m.texture_sigma_max:.1f}",
        f"Edge jitter  {m.alignment_tolerance_px:.2f} px allowed, up to {m.edge_jitter_max_px:.2f} px measured",
        f"Lighting     gain {m.lighting_gain_range[0]:.2f} – {m.lighting_gain_range[1]:.2f}",
        f"Scales       {', '.join(f'1/{s}' if s > 1 else 'full' for s in m.scales)} · {m.tiles} tiles · {m.seconds:.1f} s",
    ])


class CheckSection(QWidget):
    """Side-panel block for a check stage: level badge, issue list and metrics."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self.level_label = QLabel()
        self.level_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.issue_list = QListWidget()
        self.issue_list.setWordWrap(True)
        self.issue_list.setMinimumHeight(90)
        self.metrics_label = QLabel("")
        self.metrics_label.setObjectName("metrics")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.level_label)
        layout.addWidget(self.issue_list, 1)
        layout.addWidget(self.metrics_label)
        self.clear()

    def set_level(self, level: QualityLevel | None, word: str | None = None) -> None:
        default_word, background, foreground = LEVEL_STYLE[level]
        self.level_label.setText(f"{self._title}: {word or default_word}")
        self.level_label.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground}; font-weight: bold; "
            "border-radius: 4px; padding: 6px; }"
        )

    def set_issues(self, issues: tuple[QualityIssue, ...], ok_text: str) -> None:
        self.issue_list.clear()
        if not issues:
            item = QListWidgetItem(ok_text)
            item.setForeground(QBrush(QColor("#14532d")))
            self.issue_list.addItem(item)
        for issue in issues:
            prefix = "✖ CRITICAL" if issue.severity is Severity.CRITICAL else "⚠ Warning"
            item = QListWidgetItem(f"{prefix}: {issue.message}")
            item.setForeground(QBrush(QColor(SEVERITY_COLOURS[issue.severity])))
            item.setData(Qt.ItemDataRole.UserRole, issue.code)
            self.issue_list.addItem(item)

    def set_error(self, message: str) -> None:
        self.set_level(QualityLevel.POOR, "FAILED")
        self.issue_list.clear()
        item = QListWidgetItem(f"✖ {message}")
        item.setForeground(QBrush(QColor(SEVERITY_COLOURS[Severity.CRITICAL])))
        self.issue_list.addItem(item)
        self.metrics_label.setText("")

    def set_busy(self) -> None:
        self.set_level(None, "checking…")
        self.issue_list.clear()
        self.metrics_label.setText("")

    def clear(self) -> None:
        self.set_level(None)
        self.issue_list.clear()
        self.metrics_label.setText("")
